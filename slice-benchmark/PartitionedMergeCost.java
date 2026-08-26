import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.SortedNumericDocValuesField;
import org.apache.lucene.document.SortedSetDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeTrigger;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

/**
 * What a partitioned merge costs, per FIELD TYPE.
 *
 * <p>The question this answers is not "is this worth it for workload X" but "which formats does an
 * output have to read in full, and which can it skip past". That is a property of each format's
 * ordering, not of a use case: a format whose data is ordered by document can seek to the range an
 * output owns, and one ordered by term or by value cannot, so every output reads all of it.
 *
 * <p>Method: build one set of input segments carrying every field type, then merge those same
 * inputs repeatedly -- once into a single segment, which is the baseline an ordinary merge pays,
 * and once into each output count asked for. Bytes are counted per file extension, so each format
 * reports separately. Only the merge is counted; building the inputs and copying them are not.
 *
 * <pre>
 *   javac -cp $CORE -d /tmp/sb slice-benchmark/*.java
 *   java -cp /tmp/sb:$CORE PartitionedMergeCost -Dsegments=4 -Ddocs=5000 -Doutputs=2,4,8,16,64
 * </pre>
 */
public class PartitionedMergeCost {

    static final int SEGMENTS = Integer.getInteger("segments", 4);
    static final int DOCS = Integer.getInteger("docs", 5000);
    static final int DIMS = Integer.getInteger("dims", 64);
    static final String OUTPUTS = System.getProperty("outputs", "2,4,8,16,64");
    /**
     * hnsw or none.
     *
     * <p>Worth knowing what the vector row measures. A merge rebuilds the HNSW graph, which is many
     * random distance computations over the vectors rather than a sequential pass, so the bytes
     * read say what rebuilding costs, not what reading the output's range costs. That is the real
     * cost of merging vectors either way; it just is not comparable to the other rows, which are
     * dominated by sequential reads. Run with none to see the other formats without it.
     */
    static final String VECTORS = System.getProperty("vectors", "hnsw");

    /** Extensions grouped the way a reader thinks about them, rather than by file. */
    static final Map<String, String> FORMATS = new LinkedHashMap<>();
    static {
        FORMATS.put("stored fields", "fdt fdx fdm");
        FORMATS.put("terms dict", "tim tip tmd tmp");
        FORMATS.put("postings", "doc pos pay psm");
        FORMATS.put("norms", "nvd nvm");
        FORMATS.put("doc values", "dvd dvm dvs");
        FORMATS.put("points", "kdd kdi kdm");
        FORMATS.put("vectors", "vec vex vem veq vemf vemq");
        FORMATS.put("term vectors", "tvd tvx tvm");
        FORMATS.put("field infos", "fnm si");
    }

    public static void main(String[] args) throws Exception {
        Path root = Files.createTempDirectory("partitioned-merge-cost");
        Path inputs = root.resolve("inputs");

        try (Directory dir = FSDirectory.open(inputs)) {
            build(dir);
        }

        System.out.printf(
                Locale.ROOT,
                "inputs: %d segments x %d docs = %d docs, vectors=%s (%d dims)%n%n",
                SEGMENTS, DOCS, SEGMENTS * DOCS, VECTORS, DIMS);

        // M=1 is an ordinary merge of the same inputs: the denominator for everything else.
        Map<String, long[]> baseline = mergeOnce(root, inputs, 1);
        List<Integer> counts = new ArrayList<>();
        for (String s : OUTPUTS.split(",")) {
            counts.add(Integer.parseInt(s.trim()));
        }

        Map<Integer, Map<String, long[]>> runs = new TreeMap<>();
        for (int m : counts) {
            runs.put(m, mergeOnce(root, inputs, m));
        }

        report(baseline, runs);
    }

    /** One merge of every input segment into {@code outputs} segments, counted per extension. */
    static Map<String, long[]> mergeOnce(Path root, Path inputs, int outputs) throws IOException {
        Path work = root.resolve("m" + outputs);
        copy(inputs, work);

        ContainerIndependence.Stats st = new ContainerIndependence.Stats();
        try (Directory raw = FSDirectory.open(work);
                Directory dir = new ContainerIndependence.CountingDirectory(raw, st)) {
            IndexWriterConfig iwc = config();
            iwc.setMergePolicy(new OnePartitionedMerge(outputs));
            try (IndexWriter w = new IndexWriter(dir, iwc)) {
                // Counting starts once the writer is open, so opening it -- reading the commit
                // point and the segment infos -- is not charged to the merge.
                st.building = true;
                w.maybeMerge();
            }
            st.building = false;
        }
        Map<String, long[]> byExt = new TreeMap<>();
        for (Map.Entry<String, long[]> e : st.extRead.entrySet()) {
            byExt.computeIfAbsent(e.getKey(), k -> new long[2])[0] = e.getValue()[0];
        }
        for (Map.Entry<String, long[]> e : st.extWrite.entrySet()) {
            byExt.computeIfAbsent(e.getKey(), k -> new long[2])[1] = e.getValue()[0];
        }
        return byExt;
    }

    static void report(Map<String, long[]> baseline, Map<Integer, Map<String, long[]>> runs) {
        table("BYTES READ, and the multiple of an ordinary merge", baseline, runs, 0);
        if (VECTORS.equals("none") == false) {
            System.out.println(
                    "  (vectors: a merge rebuilds the HNSW graph, so that row is dominated by "
                            + "random-access distance computations, not by sequential reads)");
        }
        System.out.println();
        // Writes are the control: every document is written exactly once, into whichever output
        // owns it, so nothing here should move with the output count. A format that does is a
        // finding rather than a detail.
        table("BYTES WRITTEN (should not move with M)", baseline, runs, 1);
    }

    static void table(
            String title,
            Map<String, long[]> baseline,
            Map<Integer, Map<String, long[]>> runs,
            int idx) {
        System.out.println(title);
        System.out.printf(Locale.ROOT, "%-15s %-20s %12s", "format", "files", "M=1");
        for (int m : runs.keySet()) {
            System.out.printf(Locale.ROOT, " %12s %7s", "M=" + m, "ratio");
        }
        System.out.println();

        for (Map.Entry<String, String> f : FORMATS.entrySet()) {
            String[] exts = f.getValue().split(" ");
            long base = sum(baseline, exts, idx);
            if (base == 0) {
                continue; // absent from this corpus, or written by a format not listed here
            }
            System.out.printf(Locale.ROOT, "%-15s %-20s %12s", f.getKey(), f.getValue(), mb(base));
            for (Map.Entry<Integer, Map<String, long[]>> r : runs.entrySet()) {
                long v = sum(r.getValue(), exts, idx);
                System.out.printf(Locale.ROOT, " %12s %6.2fx", mb(v), (double) v / base);
            }
            System.out.println();
        }

        // Anything the grouping above missed, so a format cannot vanish from the report unnoticed.
        for (String ext : baseline.keySet()) {
            if (known(ext) == false && sum(baseline, new String[] {ext}, idx) > 0) {
                System.out.printf(
                        Locale.ROOT,
                        "%-15s %-20s %12s%n",
                        "(ungrouped)",
                        ext,
                        mb(sum(baseline, new String[] {ext}, idx)));
            }
        }
    }

    static boolean known(String ext) {
        for (String group : FORMATS.values()) {
            for (String e : group.split(" ")) {
                if (e.equals(ext)) {
                    return true;
                }
            }
        }
        return false;
    }

    static long sum(Map<String, long[]> m, String[] exts, int idx) {
        long total = 0;
        for (String e : exts) {
            long[] v = m.get(e);
            if (v != null) {
                total += v[idx];
            }
        }
        return total;
    }

    /** A unit that keeps small formats legible: a row reading 0.0 MB cannot be interpreted. */
    static String mb(long bytes) {
        if (bytes >= 1024L * 1024L) {
            return String.format(Locale.ROOT, "%.1f MB", bytes / 1024.0 / 1024.0);
        }
        if (bytes >= 1024L) {
            return String.format(Locale.ROOT, "%.1f KB", bytes / 1024.0);
        }
        return bytes + " B";
    }

    static IndexWriterConfig config() {
        IndexWriterConfig iwc = new IndexWriterConfig(new StandardAnalyzer());
        iwc.setIndexSort(new Sort(new SortField("sort", SortField.Type.STRING)));
        // Individual files, so each format reports separately rather than disappearing into a .cfs.
        iwc.setUseCompoundFile(false);
        return iwc;
    }

    static void build(Directory dir) throws IOException {
        IndexWriterConfig iwc = config();
        iwc.setMergePolicy(org.apache.lucene.index.NoMergePolicy.INSTANCE);
        Random rnd = new Random(17);
        try (IndexWriter w = new IndexWriter(dir, iwc)) {
            for (int seg = 0; seg < SEGMENTS; seg++) {
                for (int d = 0; d < DOCS; d++) {
                    w.addDocument(doc(seg * DOCS + d, rnd));
                }
                w.flush();
                w.commit();
            }
        }
    }

    static final FieldType TEXT = new FieldType(TextField.TYPE_NOT_STORED);
    static {
        TEXT.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS);
        TEXT.setStoreTermVectors(true);
        TEXT.setStoreTermVectorPositions(true);
        TEXT.setStoreTermVectorOffsets(true);
        TEXT.freeze();
    }

    /** One document carrying every field type, so no format is missing from the report. */
    static Document doc(int ord, Random rnd) {
        String id = String.format(Locale.ROOT, "id-%08d", ord);
        Document d = new Document();
        d.add(new StringField("id", id, Field.Store.NO));
        d.add(new SortedDocValuesField("sort", new BytesRef(id)));
        d.add(new StoredField("_source", source(id, rnd)));
        d.add(new Field("text", text(rnd), TEXT));
        d.add(new NumericDocValuesField("num", ord));
        d.add(new BinaryDocValuesField("bin", new BytesRef(id)));
        d.add(new SortedSetDocValuesField("sset", new BytesRef(id)));
        d.add(new SortedNumericDocValuesField("snum", ord));
        d.add(new IntPoint("point", ord));
        d.add(new LongPoint("point2d", ord, -ord));
        if (VECTORS.equals("none") == false) {
            float[] vector = new float[DIMS];
            for (int i = 0; i < DIMS; i++) {
                vector[i] = rnd.nextFloat();
            }
            d.add(new KnnFloatVectorField("vec", vector, VectorSimilarityFunction.DOT_PRODUCT));
        }
        return d;
    }

    /** Varied enough to compress like real stored content rather than to nothing. */
    static String source(String id, Random rnd) {
        StringBuilder sb = new StringBuilder(400);
        sb.append("{\"id\":\"").append(id).append("\",\"body\":\"");
        for (int i = 0; i < 40; i++) {
            sb.append(WORDS[rnd.nextInt(WORDS.length)]).append(' ');
        }
        return sb.append("\"}").toString();
    }

    /**
     * Deliberately varied in length. Norms encode a document's length, so a corpus of uniform
     * documents compresses them to almost nothing and the norms row reports on a file too small to
     * interpret -- which is exactly what a first run of this did.
     */
    static String text(Random rnd) {
        int terms = 5 + rnd.nextInt(120);
        StringBuilder sb = new StringBuilder(terms * 9);
        for (int i = 0; i < terms; i++) {
            sb.append(WORDS[rnd.nextInt(WORDS.length)]).append(' ');
        }
        return sb.toString();
    }

    static final String[] WORDS = buildVocabulary();

    /** A vocabulary large enough that the terms dictionary is not a rounding error. */
    static String[] buildVocabulary() {
        String[] words = new String[20000];
        Random rnd = new Random(3);
        for (int i = 0; i < words.length; i++) {
            StringBuilder sb = new StringBuilder(8);
            for (int c = 0; c < 8; c++) {
                sb.append((char) ('a' + rnd.nextInt(26)));
            }
            words[i] = sb.toString();
        }
        return words;
    }

    static void copy(Path from, Path to) throws IOException {
        if (Files.exists(to)) {
            try (var walk = Files.walk(to)) {
                walk.sorted(java.util.Comparator.reverseOrder()).forEach(p -> {
                    try {
                        Files.delete(p);
                    } catch (IOException e) {
                        throw new java.io.UncheckedIOException(e);
                    }
                });
            }
        }
        Files.createDirectories(to);
        try (var list = Files.list(from)) {
            for (Path p : list.toList()) {
                Files.copy(p, to.resolve(p.getFileName()));
            }
        }
    }

    /**
     * Answers one merge of everything, partitioned into a fixed number of outputs by even document
     * ranges. Even ranges are the point: the question is what a split costs per format, and where
     * the boundaries fall would only change which documents land where.
     */
    static class OnePartitionedMerge extends MergePolicy {
        final int outputs;
        boolean done;

        OnePartitionedMerge(int outputs) {
            this.outputs = outputs;
        }

        @Override
        public MergeSpecification findMerges(MergeTrigger t, SegmentInfos infos, MergeContext ctx) {
            if (done || infos.size() < 2) {
                return null;
            }
            done = true;
            List<SegmentCommitInfo> segs = new ArrayList<>();
            for (SegmentCommitInfo si : infos) {
                segs.add(si);
            }
            int[][] parts = new int[segs.size()][];
            for (int i = 0; i < segs.size(); i++) {
                int maxDoc = segs.get(i).info.maxDoc();
                int[] b = new int[outputs + 1];
                for (int o = 0; o <= outputs; o++) {
                    b[o] = (int) ((long) o * maxDoc / outputs);
                }
                parts[i] = b;
            }
            MergeSpecification spec = new MergeSpecification();
            spec.add(new Split(segs, parts, outputs));
            return spec;
        }

        @Override
        public MergeSpecification findForcedMerges(
                SegmentInfos i, int m, Map<SegmentCommitInfo, Boolean> s, MergeContext c) {
            return null;
        }

        @Override
        public MergeSpecification findForcedDeletesMerges(SegmentInfos i, MergeContext c) {
            return null;
        }
    }

    static class Split extends MergePolicy.OneMerge {
        final int[][] parts;
        final int outputs;

        Split(List<SegmentCommitInfo> segments, int[][] parts, int outputs) {
            super(segments);
            this.parts = parts;
            this.outputs = outputs;
        }

        @Override
        public boolean isPartitioned() {
            return outputs > 1;
        }

        @Override
        public int[][] getDocRangePartitions(List<CodecReader> readers) {
            return outputs > 1 ? parts : null;
        }
    }
}
