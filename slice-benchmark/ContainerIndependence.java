import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeTrigger;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Terms;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TotalHitCountCollectorManager;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefBuilder;

/**
 * Does the cost of serving one slice depend on the size of the index containing it?
 *
 * The corpus is held structurally identical and only SCALED: the same probe slice, written
 * across the same number of write epochs, inside an index that grows 1x -> 4x -> 16x. If
 * per-slice query cost is flat in index size, a slice behaves like its own small index.
 *
 * Two layouts over the identical corpus:
 *   stock  -- TieredMergePolicy, the layout we have today
 *   ranged -- each write epoch is partitioned by routing key into R ranges using the
 *             multi-output merge (R scales with the corpus, so bytes per range is constant)
 *
 * Measured per single-slice query, not simulated:
 *   bytes    -- logical bytes read through the Directory (proxy for object-store bytes)
 *   blocks   -- distinct 16 KB blocks touched  (proxy for NVMe/page-cache residency)
 *   segments -- segments whose terms dictionary is probed
 *   us       -- wall time
 *
 * Run with the patched lucene-core built from the multi-output-merge branch.
 */
public class ContainerIndependence {

    static final int BLOCK = 16 * 1024;
    static final int EPOCHS = 60;              // write epochs; a trickle slice appears in each
    static final int PROBE_DOCS = 300;        // held fixed at every scale
    static long RANGE_TARGET_BYTES = 4L << 20;
    static int SEED = 11;
    static final int WHALES = 3;
    static final int WHALE_DOCS_PER_EPOCH = 900;
    static double CAP_MB = 4;   // derived from the index/cap ratio at runtime
    static double FLUSH_MB = 2; // derived from CAP_MB so tiering has room, as in production
    /** Final size of the nomerge arm -- the common write-amplification denominator. */
    static long BASELINE_BYTES;
    /** Per-extension build reads, so the merge cost can be attributed to a format. */
    static final Map<String, Long> NOMERGE_EXT = new java.util.TreeMap<>();
    static final Map<String, Map<String, long[]>> EXT_BY_LAYOUT = new java.util.LinkedHashMap<>();
    static final Map<String, Map<String, long[]>> EXTW_BY_LAYOUT = new java.util.LinkedHashMap<>();
    /**
     * Bytes read by the nomerge arm, which performs no merges at all. Indexing a sorted index
     * reads its own stored fields back to reorder them at flush, so this floor is large and is
     * paid by every arm; merge-attributable IO is only what sits above it.
     */
    static long NOMERGE_READ, NOMERGE_WRITTEN;

    // ---------------------------------------------------------------- accounting
    static class Stats {
        long bytes;
        long written;
        long wFlush, wSplit, wTier;
        int phase; // 0=flush 1=split 2=tier
        final Set<Long> blocks = new HashSet<>();
        final Map<String, Integer> fileIds = new java.util.HashMap<>();
        /** Extension of each file id, so reads can be attributed to a format. */
        final java.util.List<String> idExt = new java.util.ArrayList<>();
        final Map<String, long[]> extRead = new java.util.TreeMap<>();
        /**
         * Bytes WRITTEN per extension, in the same four buckets as extRead. Without this the read
         * table has no denominator: a format that reads 12 GB is only alarming next to what the
         * same merges wrote, and "reads far exceed writes" is the thing being explained.
         */
        final Map<String, long[]> extWrite = new java.util.TreeMap<>();
        boolean on;
        /**
         * Bytes read while building. Bytes WRITTEN understates the cost of a partitioned merge,
         * because the merge runs once per output and rereads its inputs each time -- write
         * amplification cannot see that, so it has to be measured on the read side.
         */
        boolean building;
        long buildRead;

        int fileId(String n) {
            return fileIds.computeIfAbsent(n, k -> {
                int i = k.lastIndexOf('.');
                idExt.add(i < 0 ? "segments" : k.substring(i + 1));
                return fileIds.size();
            });
        }
        void read(int fid, long pos, long len) {
            if (building) {
                buildRead += len;
                long[] a = extRead.computeIfAbsent(idExt.get(fid), k -> new long[4]);
                a[0] += len;
                // 1=flush/other, 2=range split, 3=consolidation -- so a partitioned merge's own
                // cost can be told apart from the ordinary tiering the same arm still does.
                a[1 + phase] += len;
            }
            if (!on) return;
            bytes += len;
            long first = pos / BLOCK, last = (pos + Math.max(0, len - 1)) / BLOCK;
            for (long b = first; b <= last; b++) blocks.add((((long) fid) << 40) | b);
        }
        void reset() { bytes = 0; blocks.clear(); }
    }

    static class CountingDirectory extends FilterDirectory {
        final Stats st;
        CountingDirectory(Directory in, Stats st) { super(in); this.st = st; }
        @Override
        public org.apache.lucene.store.IndexOutput createOutput(String name, IOContext ctx)
                throws IOException {
            return new CountingOutput(in.createOutput(name, ctx), st);
        }
        @Override
        public IndexInput openInput(String name, IOContext ctx) throws IOException {
            return new CountingInput(in.openInput(name, ctx), st, st.fileId(name));
        }
    }

    /** Counts bytes written, for indexing write amplification. */
    static class CountingOutput extends org.apache.lucene.store.IndexOutput {
        final org.apache.lucene.store.IndexOutput out; final Stats st;
        CountingOutput(org.apache.lucene.store.IndexOutput out, Stats st) {
            super("counting", out.getName()); this.out = out; this.st = st;
        }
        @Override public void close() throws IOException {
            long n = out.getFilePointer();
            st.written += n;
            switch (st.phase) {
                case 1 -> st.wSplit += n;
                case 2 -> st.wTier += n;
                default -> st.wFlush += n;
            }
            String name = out.getName();
            int i = name.lastIndexOf('.');
            long[] a = st.extWrite.computeIfAbsent(
                    i < 0 ? "segments" : name.substring(i + 1), k -> new long[4]);
            a[0] += n;
            a[1 + st.phase] += n;
            out.close();
        }
        @Override public long getFilePointer() { return out.getFilePointer(); }
        @Override public long getChecksum() throws IOException { return out.getChecksum(); }
        @Override public void writeByte(byte b) throws IOException { out.writeByte(b); }
        @Override public void writeBytes(byte[] b, int off, int len) throws IOException {
            out.writeBytes(b, off, len);
        }
        @Override public void writeInt(int i) throws IOException { out.writeInt(i); }
        @Override public void writeShort(short i) throws IOException { out.writeShort(i); }
        @Override public void writeLong(long i) throws IOException { out.writeLong(i); }
    }

    /**
     * Counts logical bytes. Deliberately does NOT override readInt/readLong/readVInt so the
     * DataInput defaults decompose into counted readByte calls -- slower, but it means nothing
     * is read without being accounted for.
     */
    static class CountingInput extends IndexInput {
        final IndexInput in; final Stats st; final int fid;
        CountingInput(IndexInput in, Stats st, int fid) {
            super("counting(" + in.toString() + ")");
            this.in = in; this.st = st; this.fid = fid;
        }
        @Override public void close() throws IOException { in.close(); }
        @Override public long getFilePointer() { return in.getFilePointer(); }
        @Override public void seek(long pos) throws IOException { in.seek(pos); }
        @Override public long length() { return in.length(); }
        @Override public byte readByte() throws IOException {
            st.read(fid, in.getFilePointer(), 1); return in.readByte();
        }
        @Override public void readBytes(byte[] b, int off, int len) throws IOException {
            st.read(fid, in.getFilePointer(), len); in.readBytes(b, off, len);
        }
        @Override public IndexInput slice(String desc, long off, long len) throws IOException {
            return new CountingInput(in.slice(desc, off, len), st, fid);
        }
        @Override public IndexInput clone() {
            return new CountingInput(in.clone(), st, fid);
        }
    }

    // ---------------------------------------------------------------- per-merge attribution
    /**
     * One executed merge, with the IO it actually caused.
     *
     * The previous attribution set STATS.phase ONCE per maybeMerge() round, from a prediction
     * (wouldConsolidate) of what the policy was about to do. But maybeMerge() drains the policy in
     * a loop, so a single round can run consolidations AND splits, and every byte of both lands in
     * whichever bucket the prediction named. Anything concluded from that split is unsafe. This
     * measures each OneMerge individually instead, which is the only way to say whether a
     * partitioned merge or ordinary tiering is what reads.
     */
    record MergeIO(String kind, int inputs, long inputBytes, long read, long written) {}

    static final List<MergeIO> MERGES = new ArrayList<>();

    /**
     * Serial scheduler that brackets every merge with the read/write counters. Serial matters:
     * the counters are global, so overlapping merges would smear into each other.
     */
    static class AttributingScheduler extends org.apache.lucene.index.MergeScheduler {
        @Override
        public synchronized void merge(
                org.apache.lucene.index.MergeScheduler.MergeSource src, MergeTrigger trigger)
                throws IOException {
            while (true) {
                MergePolicy.OneMerge m = src.getNextMerge();
                if (m == null) return;
                boolean partitioned = m.isPartitioned();
                String kind = m.getClass().getSimpleName();
                if (kind.isEmpty()) kind = partitioned ? "split" : "merge";
                int inputs = m.segments.size();
                long inBytes = m.totalBytesSize();
                long r0 = STATS.buildRead, w0 = STATS.written;
                // keep the coarse buckets working, but now driven by the merge itself
                STATS.phase = partitioned ? 1 : 2;
                try {
                    src.merge(m);
                } finally {
                    STATS.phase = 0;
                    MERGES.add(new MergeIO(kind, inputs, inBytes,
                            STATS.buildRead - r0, STATS.written - w0));
                }
            }
        }

        @Override public void close() {}
    }

    // ---------------------------------------------------------------- merge policy
    /**
     * The design's rule, applied literally: any segment holding more than the range target emits
     * TWO outputs cut at the median key. Repeated application yields R ranges in O(N log R)
     * rather than the O(N*R) an R-way split would cost, because each output rescans its inputs.
     */
    static class SplitOversized extends MergePolicy {
        final int targetDocs;
        int splits, consolidations;
        /** segment name -> [min,max] routing term, refreshed by the driver each epoch */
        Map<String, BytesRef[]> ranges = new java.util.HashMap<>();

        SplitOversized(int targetDocs) { this.targetDocs = targetDocs; }

        @Override
        public MergeSpecification findMerges(MergeTrigger t, SegmentInfos infos, MergeContext ctx) {
            if (t != MergeTrigger.EXPLICIT) return null;
            MergeSpecification spec = new MergeSpecification();
            for (SegmentCommitInfo si : infos) if (ctx.getMergingSegments().contains(si)) return null;

            // 1. tiered WITHIN a range: consolidate runs whose key spans OVERLAP. Grouping by
            //    exact [min,max] equality does not work -- two runs covering the same nominal
            //    range hold different extreme terms -- so this is the adjacency rule instead:
            //    sort by min, then greedily take consecutive runs that still overlap.
            List<SegmentCommitInfo> sorted = new ArrayList<>();
            for (SegmentCommitInfo si : infos) {
                if (si.info.maxDoc() > targetDocs) continue;
                if (ranges.containsKey(si.info.name)) sorted.add(si);
            }
            sorted.sort((a, b) -> ranges.get(a.info.name)[0].compareTo(ranges.get(b.info.name)[0]));
            int i = 0;
            while (i < sorted.size()) {
                List<SegmentCommitInfo> take = new ArrayList<>();
                BytesRef groupMax = null;
                long docs = 0;
                int j = i;
                long smallest = Long.MAX_VALUE;
                while (j < sorted.size() && take.size() < 8) {
                    SegmentCommitInfo si = sorted.get(j);
                    BytesRef[] r = ranges.get(si.info.name);
                    if (!take.isEmpty() && r[0].compareTo(groupMax) > 0) break;  // no overlap
                    // SIZE-TIERED: only merge runs of comparable size. Without this a freshly
                    // merged large run immediately re-qualifies against its small neighbours,
                    // splits, and the halves re-qualify -- which is what produced 286x write
                    // amplification. Comparable-size grouping is what bounds it to L.
                    long sz = si.info.maxDoc();
                    if (!take.isEmpty() && (sz > smallest * 3L || smallest > sz * 3L)) break;
                    smallest = Math.min(smallest, sz);
                    take.add(si); docs += si.info.maxDoc();
                    if (groupMax == null || r[1].compareTo(groupMax) > 0) groupMax = r[1];
                    j++;
                }
                if (take.size() >= 4) { spec.add(new OneMerge(take)); consolidations++; i = j; }
                else i++;
            }
            if (!spec.merges.isEmpty()) return spec;

            // 2. split anything over the range target into TWO outputs.
            for (SegmentCommitInfo si : infos) {
                if (si.info.maxDoc() > targetDocs) {
                    spec.add(new Partitioned(Collections.singletonList(si), 2, null));
                    splits++;
                }
            }
            return spec.merges.isEmpty() ? null : spec;
        }
        @Override public MergeSpecification findForcedMerges(
                SegmentInfos i, int m, Map<SegmentCommitInfo, Boolean> s, MergeContext c) { return null; }
        @Override public MergeSpecification findForcedDeletesMerges(SegmentInfos i, MergeContext c) { return null; }
    }

    static class Partitioned extends MergePolicy.OneMerge {
        final int outputs; final Set<String> done;
        Partitioned(List<SegmentCommitInfo> segs, int outputs, Set<String> done) {
            super(segs); this.outputs = outputs; this.done = done;
        }
        @Override public boolean isPartitioned() { return true; }
        @Override
        public int[][] getDocRangePartitions(List<CodecReader> readers) throws IOException {
            return keyBoundaryPartitions(readers, "routing", outputs);
        }
    }

    /** Cut every totalDocs/outputs documents, on a key boundary, streaming over dictionaries. */
    static int[][] keyBoundaryPartitions(List<CodecReader> readers, String field, int outputs)
            throws IOException {
        int n = readers.size();
        int[][] starts = new int[n][];
        SortedDocValues[] dvs = new SortedDocValues[n];
        long total = 0;
        for (int i = 0; i < n; i++) {
            starts[i] = firstDocPerOrd(readers.get(i), field);
            dvs[i] = readers.get(i).getSortedDocValues(field);
            total += readers.get(i).numDocs();
        }
        long per = Math.max(1, total / outputs);
        int[] cursor = new int[n];
        List<BytesRef> cuts = new ArrayList<>();
        long acc = 0;
        while (cuts.size() < outputs - 1) {
            BytesRef min = null;
            for (int i = 0; i < n; i++) {
                if (dvs[i] == null || cursor[i] >= dvs[i].getValueCount()) continue;
                BytesRef c = dvs[i].lookupOrd(cursor[i]);
                if (min == null || c.compareTo(min) < 0) {
                    BytesRefBuilder b = new BytesRefBuilder(); b.copyBytes(c); min = b.toBytesRef();
                }
            }
            if (min == null) break;
            if (acc >= per * (cuts.size() + 1)) cuts.add(min);
            for (int i = 0; i < n; i++) {
                if (dvs[i] == null || cursor[i] >= dvs[i].getValueCount()) continue;
                if (dvs[i].lookupOrd(cursor[i]).compareTo(min) == 0) {
                    acc += starts[i][cursor[i] + 1] - starts[i][cursor[i]];
                    cursor[i]++;
                }
            }
        }
        int actual = cuts.size() + 1;
        int[][] parts = new int[n][actual + 1];
        for (int i = 0; i < n; i++) {
            parts[i][0] = 0;
            for (int c = 0; c < cuts.size(); c++) {
                parts[i][c + 1] = dvs[i] == null ? 0 : docOffsetOf(dvs[i], starts[i], cuts.get(c));
            }
            parts[i][actual] = readers.get(i).maxDoc();
            for (int o = 1; o <= actual; o++) {
                if (parts[i][o] < parts[i][o - 1]) parts[i][o] = parts[i][o - 1];
            }
        }
        return parts;
    }

    static int[] firstDocPerOrd(CodecReader r, String field) throws IOException {
        SortedDocValues dv = r.getSortedDocValues(field);
        int k = dv == null ? 0 : dv.getValueCount();
        int[] s = new int[k + 1];
        java.util.Arrays.fill(s, -1);
        if (dv != null) {
            for (int d = dv.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = dv.nextDoc()) {
                if (s[dv.ordValue()] == -1) s[dv.ordValue()] = d;
            }
        }
        s[k] = r.maxDoc();
        for (int i = k - 1; i >= 0; i--) if (s[i] == -1) s[i] = s[i + 1];
        return s;
    }

    static int docOffsetOf(SortedDocValues dv, int[] starts, BytesRef key) throws IOException {
        int ord = dv.lookupTerm(key);
        if (ord < 0) ord = -ord - 1;
        if (ord >= starts.length - 1) return starts[starts.length - 1];
        return starts[ord];
    }

    // ---------------------------------------------------------------- corpus
    /**
     * Heavy-tailed slice population, plus one probe slice of fixed size that writes a little in
     * every epoch. `scale` multiplies the number of OTHER slices only -- the probe never changes.
     */
    static Stats STATS;
    static FixedRangePolicy PE;
    static long DOCS_INDEXED;

    static String LAYOUT = "stock";

    static void build(Directory d, int scale, boolean ranged, String probe) throws IOException {
        int others = 6_000 * scale;
        int totalDocs = 0;
        Random rnd = new Random(SEED);
        int[] sizes = new int[others];
        for (int i = 0; i < others; i++) {
            sizes[i] = Math.max(1, (int) Math.min(20_000, Math.exp(rnd.nextGaussian() * 1.4 + 3.2)));
            totalDocs += sizes[i];
        }

        IndexWriterConfig iwc = new IndexWriterConfig();
        if (Boolean.parseBoolean(System.getProperty("noCFS", "false"))) {
            // Diagnostic only: compound files bundle every format into one .cfs, which hides which
            // format a read belongs to. Both arms get the same setting, and the codec has to be
            // told as well -- setUseCompoundFile only covers flushed segments, not merged ones.
            iwc.setUseCompoundFile(false);
            iwc.getCodec().compoundFormat().setShouldUseCompoundFile(false);
        }
        // (routing, _id) -- the two-level key. `seq` stands in for a k-ordered _id.
        iwc.setIndexSort(new Sort(new SortField("routing", SortField.Type.STRING),
                                  new SortField("seq", SortField.Type.LONG)));
        // What decides whether tiering has any room to work is cap/flush, not the absolute size.
        // Production is roughly 100 GB shard / 5 GB cap / ~500 MB flush -- so ~200 flushed
        // segments and ~20 cap-sized ones. A previous run left this at 256 MB, which produced
        // ONE flush per epoch at 6 MB against an 18 MB cap: stock had nothing to merge and came
        // out byte-identical to nomerge, so every stock-vs-ranged number from it was void.
        iwc.setRAMBufferSizeMB(FLUSH_MB);
        iwc.setMergeScheduler(new AttributingScheduler());
        FixedRangePolicy pe = null;
        if (LAYOUT.equals("nomerge")) { iwc.setMergePolicy(NoMergePolicy.INSTANCE); }
        else if (ranged) { pe = new FixedRangePolicy("routing", RANGE_TARGET_BYTES,
                    // NOT -Dratio: that one is the index/cap ratio in main(), and the two were
                    // sharing a name.
                    Integer.parseInt(System.getProperty("mergeRatio", "4"))); iwc.setMergePolicy(pe); }
        else {
            TieredMergePolicy tmp = new TieredMergePolicy();
            tmp.setMaxMergedSegmentMB(CAP_MB);
            // The floor rounds every small segment to the same size, which degenerates merge
            // selection at model scale. Remove it so a small index behaves like a large one.
            tmp.setFloorSegmentMB(0.01);
            iwc.setMergePolicy(tmp);
        }
        PE = pe;

        long docsSoFar = 0;
        try (IndexWriter w = new IndexWriter(d, iwc)) {
            for (int e = 0; e < EPOCHS; e++) {
                // other slices: each is written entirely within one epoch (burst) except a
                // fraction that trickles, matching a real mixed population.
                //
                // Indexed from several threads when asked, because that is not a detail: Lucene
                // buffers per thread and flushes each buffer to its OWN segment, so a real ingest
                // holds many more segments than a single-threaded one. For the hash-routed arm that
                // matters directly -- a single-tenant query touches every segment there is.
                final int epoch = e;
                if (THREADS > 1) {
                    List<Thread> pool = new ArrayList<>();
                    java.util.concurrent.atomic.AtomicInteger next =
                            new java.util.concurrent.atomic.AtomicInteger(epoch);
                    for (int t = 0; t < THREADS; t++) {
                        Thread th = new Thread(() -> {
                            try {
                                for (int i = next.getAndAdd(EPOCHS); i < others;
                                        i = next.getAndAdd(EPOCHS)) {
                                    String r = key(i);
                                    for (int j = 0; j < sizes[i]; j++) w.addDocument(doc(r));
                                }
                            } catch (IOException ex) {
                                throw new UncheckedIOException(ex);
                            }
                        });
                        th.start();
                        pool.add(th);
                    }
                    for (Thread th : pool) {
                        try { th.join(); } catch (InterruptedException ignore) { }
                    }
                    for (int i = epoch; i < others; i += EPOCHS) docsSoFar += sizes[i];
                } else {
                    for (int i = e; i < others; i += EPOCHS) {
                        String r = key(i);
                        for (int j = 0; j < sizes[i]; j++) { w.addDocument(doc(r)); docsSoFar++; }
                    }
                }
                // the probe slice writes a slice of its documents in every epoch (trickle)
                for (int j = 0; j < PROBE_DOCS / EPOCHS; j++) { w.addDocument(doc(probe)); docsSoFar++; }
                // Whales: continuously-written large tenants. These MUST be indexed -- an earlier
                // version probed key(900_000), which no epoch ever wrote, so every whale figure it
                // reported was the cost of a miss.
                for (int wi = 0; wi < WHALES; wi++) {
                    String wk = whaleKey(wi);
                    if (wi == DELETED_WHALE && e > WHALE_DELETE_EPOCH) continue;   // it is gone
                    for (int j = 0; j < WHALE_DOCS_PER_EPOCH; j++) { w.addDocument(doc(wk)); docsSoFar++; }
                }
                // Tenants leave. Everything measured before this existed was a purely growing
                // index, which is the one shape where a partition never has to give ground -- and
                // it is not the shape the design's hardest claims are about.
                if (CHURN_PCT > 0 && e >= CHURN_LAG) {
                    for (int i = e - CHURN_LAG; i < others; i += EPOCHS) {
                        if (Math.floorMod(i * 0x9E3779B1, 100) >= CHURN_PCT) continue;
                        w.deleteDocuments(new Term("routing", key(i)));
                        DELETED_SLICES++;
                    }
                }
                // A large tenant leaving is the case the design makes its sharpest claim about:
                // it owns whole segments, so the deletion should be a file removal rather than a
                // rewrite. Deleting whale 2 leaves whale 0 as the probe.
                if (DELETED_WHALE >= 0 && e == WHALE_DELETE_EPOCH) {
                    w.deleteDocuments(new Term("routing", whaleKey(DELETED_WHALE)));
                }
                w.flush();
                if (pe != null) {
                    pe.setDepthFor(docsSoFar);
                    for (int round = 0; round < MERGE_ROUNDS; round++) {
                        refreshExtremes(w, pe);
                        int bs = pe.splits + pe.idSplits, bm = pe.merges;
                        // Attribution now happens per merge, inside AttributingScheduler. A round
                        // can drain both rules, so predicting one bucket for the whole round was
                        // wrong.
                        w.maybeMerge();
                        if (pe.splits + pe.idSplits == bs && pe.merges == bm) break;
                    }
                }
            }
            // Let the merge policy reach its steady state before anything is measured. Without
            // this the final segment count is whatever happened to be outstanding when ingest
            // stopped, which is a property of when we stopped rather than of the policy -- and it
            // flatters or penalises an arm depending on how far behind its merges were. Both arms
            // are drained the same way.
            for (int settle = 0; settle < 200; settle++) {
                int before = MERGES.size();
                if (pe != null) refreshExtremes(w, pe);
                w.maybeMerge();
                if (MERGES.size() == before) break;
            }
            w.commit();
        }
        DOCS_INDEXED = docsSoFar;
    }

    /**
     * Per segment: the min/max of BOTH levels of the key -- routing hash, then `seq`, which stands
     * in for `_id`. A segment attribute in a real implementation. The second level is what names a
     * whale's sub-ranges: once every document in a segment carries the same routing value, the
     * first level cannot tell two of its ranges apart, and a range that cannot be named is a range
     * that consolidation merges straight back together.
     *
     * <p>The index is sorted by `(routing, seq)`, so within a segment holding one routing value the
     * seq values are monotonic and the extremes are the first and last document.
     */
    static void refreshExtremes(IndexWriter w, FixedRangePolicy pe) throws IOException {
        try (DirectoryReader r = DirectoryReader.open(w)) {
            Map<String, long[]> m = new java.util.HashMap<>();
            for (LeafReaderContext c : r.leaves()) {
                Terms t = c.reader().terms("routing");
                if (t == null) continue;
                String name = ((org.apache.lucene.index.SegmentReader)
                        org.apache.lucene.index.FilterLeafReader.unwrap(c.reader()))
                        .getSegmentName();
                long minSeq = 0, maxSeq = 0;
                org.apache.lucene.index.NumericDocValues seq = c.reader().getNumericDocValues("seq");
                if (seq != null && seq.advanceExact(0)) {
                    minSeq = maxSeq = seq.longValue();
                    org.apache.lucene.index.NumericDocValues last =
                            c.reader().getNumericDocValues("seq");
                    if (last.advanceExact(c.reader().maxDoc() - 1)) maxSeq = last.longValue();
                }
                m.put(name, new long[]{Long.parseLong(t.getMin().utf8ToString(), 16),
                                       Long.parseLong(t.getMax().utf8ToString(), 16),
                                       minSeq, maxSeq});
            }
            pe.extremes = m;
        }
    }

    static final int MERGE_ROUNDS = Integer.parseInt(System.getProperty("rounds", "8"));
    static final boolean TEXT = Boolean.parseBoolean(System.getProperty("text", "false"));
    /** Percentage of ordinary slices that are deleted during the run, and how long they live. */
    /** Indexing threads. Lucene buffers and flushes per thread, so this drives segment count. */
    static final int THREADS = Integer.getInteger("threads", 1);
    static final int CHURN_PCT = Integer.getInteger("churn", 0);
    static final int CHURN_LAG = Integer.getInteger("churnLag", 10);
    /** Which whale is deleted mid-run, or -1 for none. Whale 0 stays: it is the probe. */
    static final int DELETED_WHALE = Integer.getInteger("deleteWhale", -1);
    static final int WHALE_DELETE_EPOCH = EPOCHS / 2;
    static int DELETED_SLICES;
    /** Whether documents carry a point field, which a partitioned merge reads once per output. */
    static final boolean POINTS = Boolean.parseBoolean(System.getProperty("points", "false"));
    /**
     * Per thread, so that concurrent indexing produces the same documents without contending on one
     * generator. Seeded from the thread's index, so a run is reproducible for a given thread count.
     */
    static final ThreadLocal<Random> TERM_RND = ThreadLocal.withInitial(() -> new Random(5));
    static final ThreadLocal<Random> SRC_RND = ThreadLocal.withInitial(() -> new Random(17));
    /**
     * Vocabulary for the stored payload. A previous version stored the same 400 identical
     * characters in every document, which compresses to almost nothing -- so stored fields were a
     * rounding error in the index and no measurement of them meant anything. Real _source is varied
     * and is usually the largest thing in the index, so it is generated with real entropy here.
     */
    static final String[] WORDS = new String[2048];
    static {
        Random r = new Random(99);
        for (int i = 0; i < WORDS.length; i++) {
            StringBuilder w = new StringBuilder();
            int len = 3 + r.nextInt(8);
            for (int c = 0; c < len; c++) w.append((char) ('a' + r.nextInt(26)));
            WORDS[i] = w.toString();
        }
    }

    /** A document's stored payload, shaped like the JSON _source an ES document would carry. */
    static String source(String routing, long seq) {
        StringBuilder sb = new StringBuilder(448);
        sb.append("{\"tenant\":\"").append(routing).append("\",\"seq\":").append(seq);
        sb.append(",\"ts\":").append(1_700_000_000L + seq * 37L);
        sb.append(",\"user\":\"u").append(SRC_RND.get().nextInt(100_000)).append('\"');
        sb.append(",\"path\":\"/api/v").append(SRC_RND.get().nextInt(3)).append('/')
          .append(WORDS[SRC_RND.get().nextInt(WORDS.length)]).append('/')
          .append(WORDS[SRC_RND.get().nextInt(WORDS.length)]).append('\"');
        sb.append(",\"msg\":\"");
        for (int i = 0; i < 14; i++) sb.append(WORDS[SRC_RND.get().nextInt(WORDS.length)]).append(' ');
        sb.append("\",\"bytes\":").append(SRC_RND.get().nextInt(1_000_000));
        sb.append(",\"ok\":").append(SRC_RND.get().nextBoolean()).append('}');
        return sb.toString();
    }
    static final String[] PAD = new String[64];
    static { for (int i = 0; i < PAD.length; i++) PAD[i] = "p" + i; }

    static String key(int i) {
        int h = i * 0x9E3779B1;                 // uniform 32-bit spread
        h ^= (h >>> 16); h *= 0x7feb352d; h ^= (h >>> 15);
        return String.format(Locale.ROOT, "%08x", h);
    }

    /** Whale ids live above the regular slice population so they never collide with it. */
    static String whaleKey(int i) { return key(900_000 + i); }

    static final java.util.concurrent.atomic.AtomicLong SEQ_GEN =
            new java.util.concurrent.atomic.AtomicLong();

    static Document doc(String routing) {
        Document d = new Document();
        final long seq = SEQ_GEN.getAndIncrement();
        d.add(new org.apache.lucene.document.NumericDocValuesField("seq", seq));
        d.add(new StringField("routing", routing, Field.Store.NO));
        d.add(new SortedDocValuesField("routing", new BytesRef(routing)));
        d.add(new NumericDocValuesField("v", 1));
        d.add(new StringField("pad", PAD[Math.abs(routing.hashCode()) % PAD.length], Field.Store.NO));
        if (TEXT) {
            // High-cardinality terms are what make a terms-dictionary merge expensive, and the
            // terms dictionary is the one format a partitioned merge must walk once per output.
            StringBuilder sb = new StringBuilder();
            for (int t = 0; t < 12; t++) sb.append("t").append(TERM_RND.get().nextInt(200_000)).append(' ');
            d.add(new org.apache.lucene.document.TextField("body", sb.toString(), Field.Store.NO));
        }
        d.add(new org.apache.lucene.document.StoredField("_source", source(routing, seq)));
        if (POINTS) {
            // A partitioned merge reads a block k-d tree once per output, exactly as it did the
            // terms dictionary before the single pass: the tree is ordered by value, so an output
            // cannot seek past the documents it does not own. Every ES index has at least a
            // @timestamp, so a corpus without one understates the merge cost.
            d.add(new org.apache.lucene.document.LongPoint("ts", 1_700_000_000L + seq * 37L));
        }
        return d;
    }

    // ---------------------------------------------------------------- measurement
    static double STRADDLE_PCT, STRADDLE_L0_PCT;
    static int STRADDLERS;
    static double BEST_IMBALANCE;
    static int CANDIDATES;
    static long IDX_BYTES, IDX_BLOCKS, SRI_KB;
    static double IDX_SEGS, IDX_DOCS_PER_SEG, IDX_US;

    record Result(long bytes, long blocks, double segments, double micros, int totalSegs, long sizeMb) {}

    /**
     * ONE query path, used by every layout. Skipping a segment whose [min,max] cannot contain the
     * key is available to any layout -- it is a segment attribute, not a feature of range merging.
     * Stock simply never benefits from it, because its segments span the whole key space, and that
     * is the result rather than an artefact of measuring the two arms differently. An earlier
     * version ran the ranged arm through a raw TermsEnum and the stock arm through a full
     * IndexSearcher, which compared two different amounts of work; every number it produced has
     * been withdrawn.
     */
    static long runProbe(DirectoryReader r, BytesRef[] mins, BytesRef[] maxs, BytesRef target)
            throws IOException {
        long probed = 0;
        for (int L = 0; L < mins.length; L++) {
            if (mins[L] == null
                    || target.compareTo(mins[L]) < 0
                    || target.compareTo(maxs[L]) > 0) continue;
            probed++;
            Terms t = r.leaves().get(L).reader().terms("routing");
            TermsEnum te = t.iterator();
            if (te.seekExact(target) == false) continue;
            // Retrieve, do not just count: a single-tenant query reads its postings.
            PostingsEnum pe = te.postings(null, PostingsEnum.NONE);
            while (pe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) { /* consume */ }
        }
        return probed;
    }

    static Result probe(Directory d, Stats st, String probeKey, boolean prune) throws IOException {
        try (DirectoryReader r = DirectoryReader.open(d)) {
            // Warm: build the per-segment [min,max] the design stores as a segment attribute.
            BytesRef[] mins = new BytesRef[r.leaves().size()], maxs = new BytesRef[r.leaves().size()];
            for (int i = 0; i < r.leaves().size(); i++) {
                Terms t = r.leaves().get(i).reader().terms("routing");
                mins[i] = t == null ? null : BytesRef.deepCopyOf(t.getMin());
                maxs[i] = t == null ? null : BytesRef.deepCopyOf(t.getMax());
            }
            BytesRef target = new BytesRef(probeKey);
            // untimed warm-up, through the SAME path that is later timed
            for (int i = 0; i < 3; i++) runProbe(r, mins, maxs, target);

            st.reset(); st.on = true;
            long probed = 0;
            int iters = 30;
            long t0 = System.nanoTime();
            for (int it = 0; it < iters; it++) probed += runProbe(r, mins, maxs, target);
            long ns = System.nanoTime() - t0;
            st.on = false;
            long size = 0;
            for (String f : d.listAll()) size += d.fileLength(f);
            // Straddle cost: docs in segments spanning the boundary a controller would actually
            // pick. That is the DOCUMENT median, not the midpoint of the hash space -- slice sizes
            // are lognormal, so document mass is not uniform over the key space and the previous
            // hardcoded 0x80000000 measured a boundary nobody would choose.
            long totalDocs = 0;
            java.util.List<long[]> byKey = new java.util.ArrayList<>();   // {minKey, maxKey, docs}
            for (LeafReaderContext c : r.leaves()) {
                totalDocs += c.reader().maxDoc();
                Terms t = c.reader().terms("routing");
                if (t == null) continue;
                try {
                    byKey.add(new long[]{Long.parseLong(t.getMin().utf8ToString(), 16),
                                         Long.parseLong(t.getMax().utf8ToString(), 16),
                                         c.reader().maxDoc()});
                } catch (NumberFormatException ignore) { }
            }
            byKey.sort((a, b) -> Long.compare(a[0], b[0]));
            long half = totalDocs / 2, run = 0, medianKey = 0;
            for (long[] x : byKey) { run += x[2]; if (run >= half) { medianKey = x[0]; break; } }
            // What straddles, and at which depth it was written. A segment that has never been
            // range-partitioned spans the whole key space and straddles every boundary, so it is
            // worth separating from the older, coarser segments: the first is what splitting at
            // FLUSH time would remove, the second is what only re-cutting could.
            long straddle = 0, straddleL0 = 0;
            int straddlers = 0;
            for (long[] x : byKey) {
                if (x[0] >= medianKey || x[1] < medianKey) continue;
                straddle += x[2];
                straddlers++;
                long diff = x[0] ^ x[1];
                int segDepth = diff == 0 ? 32 : Math.min(32, Long.numberOfLeadingZeros(diff) - 32);
                if (segDepth == 0) straddleL0 += x[2];
            }
            STRADDLE_PCT = totalDocs == 0 ? 0 : (100.0 * straddle / totalDocs);
            STRADDLE_L0_PCT = totalDocs == 0 ? 0 : (100.0 * straddleL0 / totalDocs);
            STRADDLERS = straddlers;

            // Balance: sort segments by their min key, then test every segment edge as a
            // candidate shard boundary. The best one is what the controller would pick.
            java.util.List<long[]> segs = new java.util.ArrayList<>();   // {minKey, docs}
            for (LeafReaderContext c : r.leaves()) {
                Terms t = c.reader().terms("routing");
                if (t == null) continue;
                try {
                    segs.add(new long[]{Long.parseLong(t.getMin().utf8ToString(), 16),
                                        c.reader().maxDoc()});
                } catch (NumberFormatException ignore) { }
            }
            segs.sort((a, b) -> Long.compare(a[0], b[0]));
            long tot = 0; for (long[] x : segs) tot += x[1];
            double best = 1.0; long acc = 0;
            for (int i = 0; i < segs.size() - 1; i++) {
                acc += segs.get(i)[1];
                best = Math.min(best, Math.abs(acc / (double) tot - 0.5) * 2);
            }
            BEST_IMBALANCE = segs.size() < 2 ? Double.NaN : best * 100;
            CANDIDATES = Math.max(0, segs.size() - 1);
            measureRangeBalance(r);
            return new Result(st.bytes / iters, st.blocks.size(), probed / (double) iters,
                    ns / 1000.0 / iters, r.leaves().size(), size / (1024 * 1024));
        }
    }


    // ---------------------------------------------------------------- range balance
    /** Ranges at the end of the build, and how unevenly the data sits across them. */
    static int RANGES;
    static long LEFTOVER_DELETES;
    static double RANGE_MAX_OVER_MEAN, RANGE_P95_OVER_MEDIAN, RANGE_TOP_DECILE_SHARE;
    static double RANGE_MAX_OVER_TARGET;

    /**
     * Hashing makes ranges equal in tenant COUNT, which says nothing about bytes: tenant sizes are
     * heavy-tailed, so a range holding a large tenant carries many times its share. This is the
     * measurement that says whether refining ranges individually is worth its write amplification,
     * so it reports the spread rather than an average -- an average would hide exactly the case in
     * question.
     *
     * <p>A range is named the way the merge policy names it: the bits a segment's min and max
     * routing keys agree on. Segments in the same range are summed.
     */
    static void measureRangeBalance(DirectoryReader r) throws IOException {
        Map<String, Long> docsPerRange = new java.util.HashMap<>();
        for (LeafReaderContext c : r.leaves()) {
            Terms t = c.reader().terms("routing");
            if (t == null) continue;
            long min, max;
            try {
                min = Long.parseLong(t.getMin().utf8ToString(), 16);
                max = Long.parseLong(t.getMax().utf8ToString(), 16);
            } catch (NumberFormatException ignore) { continue; }
            long diff = min ^ max;
            int d = diff == 0 ? 32 : Math.min(32, Long.numberOfLeadingZeros(diff) - 32);
            String key = d + ":" + Long.toHexString(d == 0 ? 0 : (min >>> (32 - d)));
            if (min == max) {
                // A whale's sub-ranges are named on the second level, exactly as the merge policy
                // names them; counting them as one range would hide the thing being measured.
                org.apache.lucene.index.NumericDocValues seq =
                        c.reader().getNumericDocValues("seq");
                if (seq != null && seq.advanceExact(0)) {
                    long lo = seq.longValue(), hi = lo;
                    org.apache.lucene.index.NumericDocValues last =
                            c.reader().getNumericDocValues("seq");
                    if (last.advanceExact(c.reader().maxDoc() - 1)) hi = last.longValue();
                    long sdiff = lo ^ hi;
                    int sd = sdiff == 0 ? 64 : Long.numberOfLeadingZeros(sdiff);
                    key += "/" + sd + ":" + Long.toHexString(sd == 0 ? 0 : (lo >>> (64 - sd)));
                }
            }
            docsPerRange.merge(key, (long) c.reader().maxDoc(), Long::sum);
        }
        LEFTOVER_DELETES = r.maxDoc() - r.numDocs();
        RANGES = docsPerRange.size();
        if (RANGES == 0) {
            RANGE_MAX_OVER_MEAN = RANGE_P95_OVER_MEDIAN = RANGE_TOP_DECILE_SHARE = Double.NaN;
            return;
        }
        long[] sizes = docsPerRange.values().stream().mapToLong(Long::longValue).sorted().toArray();
        long total = 0;
        for (long v : sizes) total += v;
        double mean = total / (double) RANGES;
        double median = sizes[RANGES / 2];
        RANGE_MAX_OVER_MEAN = sizes[RANGES - 1] / mean;
        RANGE_P95_OVER_MEDIAN = median == 0 ? Double.NaN : sizes[(int) (RANGES * 0.95)] / median;
        long top = 0;
        for (int i = Math.max(0, RANGES - Math.max(1, RANGES / 10)); i < RANGES; i++) top += sizes[i];
        RANGE_TOP_DECILE_SHARE = 100.0 * top / total;
        // The guarantee refinement actually offers is an upper bound per range, not equality, so
        // this is the number that says whether it held. Bytes per document are estimated from the
        // index as a whole -- ranges differ in document count, not in document shape.
        long bytes = 0;
        try {
            for (String f : r.directory().listAll()) bytes += r.directory().fileLength(f);
        } catch (IOException ignore) { }
        double perDoc = total == 0 ? 0 : bytes / (double) total;
        RANGE_MAX_OVER_TARGET = sizes[RANGES - 1] * perDoc / RANGE_TARGET_BYTES;
    }

    public static void main(String[] args) throws Exception {
        int[] scales = {Integer.parseInt(System.getProperty("scale", "16"))};
        double ratio = Double.parseDouble(System.getProperty("ratio", "20"));
        // Must match the corpus actually being built, or cap and range target are both wrong and
        // the run silently measures a different regime. Text roughly septuples the index.
        double indexMb = Double.parseDouble(System.getProperty("indexMb",
                Boolean.parseBoolean(System.getProperty("text", "false")) ? "378" : "56"));
        CAP_MB = indexMb / ratio;
        RANGE_TARGET_BYTES = (long) (CAP_MB * 1024 * 1024
                * Double.parseDouble(System.getProperty("targetMult", "1")));
        // ~10 flushed segments per cap-sized segment, as in production.
        FLUSH_MB = Math.max(0.5, CAP_MB / Double.parseDouble(System.getProperty("flushPerCap", "10")));
        System.out.printf(Locale.ROOT, "index/cap ratio = %.0f  ->  cap = %.2f MB, "
                + "range target = %.2f MB, flush = %.2f MB (%.0f flushes/cap), floor removed%n",
                ratio, CAP_MB, CAP_MB, FLUSH_MB, CAP_MB / FLUSH_MB);
        String probeKey = key(42);   // same hex space as every other slice
        String whaleKey = whaleKey(0);
        System.out.printf(Locale.ROOT, "probe slice = %s, %d docs, written across %d epochs%n",
                probeKey, PROBE_DOCS, EPOCHS);
        System.out.printf(Locale.ROOT, "%n%-8s %-8s %8s %8s %10s %10s %9s %9s %9s %9s %11s%n",
                "layout", "scale", "index MB", "segs", "bytes/q", "blocks", "segs/q", "us/q",
                "write amp", "read amp", "straddle %");
        SEED = Integer.parseInt(System.getProperty("seed", "11"));
        for (int scale : scales) {
            for (String layout : new String[]{"nomerge", "stock", "ranged"}) {
                LAYOUT = layout;
                boolean ranged = layout.equals("ranged");
                Path dir = Files.createTempDirectory("ci");
                try {
                    Stats st = new Stats();
                    try (Directory fs = FSDirectory.open(dir)) {
                        STATS = st;
                        Directory bd = new CountingDirectory(fs, st);
                        long buildT0 = System.nanoTime();
                        st.building = true; st.buildRead = 0;
                        MERGES.clear();
                        DELETED_SLICES = 0;
                        build(bd, scale, ranged, probeKey);   // st.on=false: writes counted, reads not
                        long buildMs = (System.nanoTime() - buildT0) / 1_000_000;
                        st.building = false;
                        long written = st.written;
                        long readDuringBuild = st.buildRead;
                        Directory d = new CountingDirectory(fs, st);
                        Result res = probe(d, st, probeKey, ranged);
                        Result wres = probe(d, st, whaleKey, ranged);
                        // nomerge runs first and defines the logical corpus size. Normalising each
                        // arm by its OWN final size (as before) rewards an arm for writing a bigger
                        // index, so the arms were not comparable.
                        if (layout.equals("nomerge")) {
                            BASELINE_BYTES = res.sizeMb() * 1024L * 1024L;
                            NOMERGE_READ = readDuringBuild;
                            NOMERGE_WRITTEN = written;
                            NOMERGE_EXT.clear();
                            for (Map.Entry<String, long[]> en : st.extRead.entrySet()) {
                                NOMERGE_EXT.put(en.getKey(), en.getValue()[0]);
                            }
                        }
                        EXT_BY_LAYOUT.put(layout, new java.util.TreeMap<>(st.extRead));
                        EXTW_BY_LAYOUT.put(layout, new java.util.TreeMap<>(st.extWrite));
                        double wamp = written / (double) Math.max(1, BASELINE_BYTES);
                        // Merge-attributable IO only: everything above what a no-merge build of the
                        // same corpus already pays (flush, and the read-back a sorted index does to
                        // reorder its own stored fields).
                        long mergeRead = Math.max(0, readDuringBuild - NOMERGE_READ);
                        long mergeWritten = Math.max(0, written - NOMERGE_WRITTEN);
                        double ramp = mergeRead / (double) Math.max(1, BASELINE_BYTES);
                        System.out.printf(Locale.ROOT,
                                "%-8s %-8s %8d %8d %10d %10d %9.1f %9.0f %9.2f %9.2f %11.1f%n",
                                layout, scale + "x", res.sizeMb(),
                                res.totalSegs(), res.bytes(), res.blocks(), res.segments(),
                                res.micros(), wamp, ramp, STRADDLE_PCT);
                        System.out.printf(Locale.ROOT,
                                "  └─ build %,d ms | shard split: %d candidate boundaries, "
                                + "best imbalance %.2f%%%n", buildMs, CANDIDATES, BEST_IMBALANCE);
                        System.out.printf(Locale.ROOT,
                                "  └─ straddle at the split boundary: %.1f%% of the index in "
                                + "%d segments, of which %.1f%% is never-partitioned L0%n",
                                STRADDLE_PCT, STRADDLERS, STRADDLE_L0_PCT);
                        System.out.printf(Locale.ROOT,
                                "  └─ churn: %,d slices deleted | %,d docs still deleted in the "
                                + "index (%.1f%% of it)%n",
                                DELETED_SLICES, LEFTOVER_DELETES,
                                100.0 * LEFTOVER_DELETES / Math.max(1, DOCS_INDEXED));
                        System.out.printf(Locale.ROOT,
                                "  └─ ranges: %d | docs per range max/mean %.2fx, p95/median %.2fx,"
                                + " top decile holds %.1f%%, biggest range %.2fx target%n",
                                RANGES, RANGE_MAX_OVER_MEAN, RANGE_P95_OVER_MEDIAN,
                                RANGE_TOP_DECILE_SHARE, RANGE_MAX_OVER_TARGET);
                        if (PE != null) {
                            System.out.printf(Locale.ROOT,
                                "  └─ partition: %d merges, %,d input docs vs %,d indexed (%.2fx), "
                                + "%,d outputs total, k=%d | splits %d, id-splits %d, "
                                + "consolidations %d (of which %d re-compact L0, %,d MB), "
                                + "reclaims %d, absorbs %d%n",
                                PE.partitionMerges, PE.partitionInputDocs, DOCS_INDEXED,
                                PE.partitionInputDocs / (double) Math.max(1, DOCS_INDEXED),
                                PE.partitionOutputs, 1 << PE.depth,
                                PE.splits, PE.idSplits, PE.merges, PE.l0Consolidations,
                                PE.l0ConsolidationBytes >> 20, PE.reclaims, PE.absorbs);
                        }
                        long tot = Math.max(1, st.wFlush + st.wSplit + st.wTier);
                        System.out.printf(Locale.ROOT,
                                "  └─ writes: flush %.0f%%  split/descend %.0f%%  tiering %.0f%%%n",
                                100.0 * st.wFlush / tot, 100.0 * st.wSplit / tot,
                                100.0 * st.wTier / tot);
                        System.out.printf(Locale.ROOT,
                                "  └─ whale (%,d docs): %d bytes/q, %.1f segs/q, %.0f us/q%n",
                                WHALE_DOCS_PER_EPOCH * EPOCHS,
                                wres.bytes(), wres.segments(), wres.micros());
                        System.out.printf(Locale.ROOT,
                                "  └─ build IO: %,d MB written / %,d MB read  |  MERGE only: "
                                + "%,d MB written, %,d MB read%n",
                                written >> 20, readDuringBuild >> 20,
                                mergeWritten >> 20, mergeRead >> 20);
                        printMergeBreakdown();
                    }
                } finally {
                    try (var w = Files.walk(dir)) {
                        w.sorted(Collections.reverseOrder()).forEach(p -> p.toFile().delete());
                    }
                }
            }
            printMergeReadsByFormat();
        }
    }

    /**
     * Reads and writes of every executed merge, grouped by the OneMerge subclass that produced it.
     *
     * read/write is the number the whole question turns on. An ordinary merge copies its inputs
     * once, so it should sit at ~1. Anything materially above 1 is a merge reading data it does not
     * write -- either the same bytes more than once, or bytes it discards.
     * read/in is the same thing normalised by the merge's input size, which for a partitioned merge
     * should approach the number of outputs if each output rescans the inputs.
     */
    static void printMergeBreakdown() {
        if (MERGES.isEmpty()) return;
        Map<String, long[]> byKind = new java.util.TreeMap<>();
        for (MergeIO m : MERGES) {
            long[] a = byKind.computeIfAbsent(m.kind(), k -> new long[4]);
            a[0]++; a[1] += m.inputBytes(); a[2] += m.read(); a[3] += m.written();
        }
        System.out.printf(Locale.ROOT, "     %-16s %7s %10s %10s %10s %8s %8s%n",
                "merge kind", "count", "in MB", "read MB", "write MB", "read/wr", "read/in");
        long[] tot = new long[4];
        for (Map.Entry<String, long[]> e : byKind.entrySet()) {
            long[] a = e.getValue();
            for (int i = 0; i < 4; i++) tot[i] += a[i];
            System.out.printf(Locale.ROOT, "     %-16s %7d %10.1f %10.1f %10.1f %8.2f %8.2f%n",
                    e.getKey(), a[0], a[1] / 1048576.0, a[2] / 1048576.0, a[3] / 1048576.0,
                    a[3] == 0 ? 0 : a[2] / (double) a[3], a[1] == 0 ? 0 : a[2] / (double) a[1]);
        }
        System.out.printf(Locale.ROOT, "     %-16s %7d %10.1f %10.1f %10.1f %8.2f %8.2f%n",
                "ALL MERGES", tot[0], tot[1] / 1048576.0, tot[2] / 1048576.0, tot[3] / 1048576.0,
                tot[3] == 0 ? 0 : tot[2] / (double) tot[3], tot[1] == 0 ? 0 : tot[2] / (double) tot[1]);
    }

    /**
     * Where a partitioned merge's extra reads actually go. Only meaningful with -DnoCFS=true:
     * otherwise every format shares one .cfs and every read is attributed to it.
     */
    static void printMergeReadsByFormat() {
        Map<String, long[]> stock = EXT_BY_LAYOUT.get("stock");
        Map<String, long[]> ranged = EXT_BY_LAYOUT.get("ranged");
        if (stock == null || ranged == null) {
            return;
        }
        java.util.Set<String> exts = new java.util.TreeSet<>();
        exts.addAll(stock.keySet());
        exts.addAll(ranged.keySet());
        System.out.printf(Locale.ROOT, "%nmerge reads by format (MB, above the no-merge build)%n");
        System.out.printf(Locale.ROOT, "%-10s %12s %12s %10s   %s%n",
                "ext", "stock", "ranged", "ranged/stock", "what it is");
        long totS = 0, totR = 0;
        for (String e : exts) {
            long base = NOMERGE_EXT.getOrDefault(e, 0L);
            long s0 = Math.max(0, (stock.containsKey(e) ? stock.get(e)[0] : 0) - base);
            long r0 = Math.max(0, (ranged.containsKey(e) ? ranged.get(e)[0] : 0) - base);
            totS += s0;
            totR += r0;
            if (s0 + r0 < (1 << 20)) {
                continue;
            }
            System.out.printf(Locale.ROOT, "%-10s %12.1f %12.1f %10s   %s%n",
                    e, s0 / 1048576.0, r0 / 1048576.0,
                    s0 == 0 ? "-" : String.format(Locale.ROOT, "%.2fx", r0 / (double) s0),
                    describe(e));
        }
        System.out.printf(Locale.ROOT, "%-10s %12.1f %12.1f %10s%n", "TOTAL",
                totS / 1048576.0, totR / 1048576.0,
                totS == 0 ? "-" : String.format(Locale.ROOT, "%.2fx", totR / (double) totS));

        System.out.printf(Locale.ROOT, "%nranged reads by what triggered them (MB)%n");
        System.out.printf(Locale.ROOT, "%-10s %12s %12s %12s%n",
                "ext", "flush/other", "range split", "consolidation");
        for (String e : exts) {
            long[] a = ranged.get(e);
            if (a == null || a[0] < (1 << 20)) {
                continue;
            }
            System.out.printf(Locale.ROOT, "%-10s %12.1f %12.1f %12.1f%n",
                    e, a[1] / 1048576.0, a[2] / 1048576.0, a[3] / 1048576.0);
        }

        // The same table with its denominator: what those merges WROTE. A format whose split
        // column reads far more than it writes is being rescanned per output.
        for (String layout : new String[]{"stock", "ranged"}) {
            Map<String, long[]> rd = EXT_BY_LAYOUT.get(layout);
            Map<String, long[]> wr = EXTW_BY_LAYOUT.get(layout);
            if (rd == null || wr == null) continue;
            System.out.printf(Locale.ROOT,
                    "%n%s: merge IO per format (MB) -- split and consolidation only, "
                    + "flush excluded%n", layout);
            System.out.printf(Locale.ROOT, "%-10s %10s %10s %8s   %10s %10s %8s%n",
                    "ext", "split rd", "split wr", "rd/wr", "consol rd", "consol wr", "rd/wr");
            for (String e : exts) {
                long[] r = rd.get(e);
                long[] w = wr.get(e);
                if (r == null && w == null) continue;
                long sr = r == null ? 0 : r[2], sw = w == null ? 0 : w[2];
                long cr = r == null ? 0 : r[3], cw = w == null ? 0 : w[3];
                if (sr + sw + cr + cw < (1 << 20)) continue;
                System.out.printf(Locale.ROOT,
                        "%-10s %10.1f %10.1f %8s   %10.1f %10.1f %8s%n",
                        e, sr / 1048576.0, sw / 1048576.0,
                        sw == 0 ? "-" : String.format(Locale.ROOT, "%.2f", sr / (double) sw),
                        cr / 1048576.0, cw / 1048576.0,
                        cw == 0 ? "-" : String.format(Locale.ROOT, "%.2f", cr / (double) cw));
            }
        }
    }

    static String describe(String ext) {
        return switch (ext) {
            case "tim", "tip", "tmd" -> "terms dictionary";
            case "doc", "pos", "pay" -> "postings";
            case "fdt", "fdx", "fdm" -> "stored fields";
            case "dvd", "dvm" -> "doc values";
            case "nvd", "nvm" -> "norms";
            case "kdd", "kdi", "kdm" -> "points";
            case "fnm" -> "field infos";
            case "si", "segments" -> "segment metadata";
            case "cfs", "cfe" -> "compound (run with -DnoCFS=true to split this out)";
            default -> "";
        };
    }
}
