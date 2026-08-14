import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeTrigger;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;

/**
 * Range partitioning on FIXED subdivisions of the routing hash space.
 *
 * The routing value is the zero-padded hex of a 32-bit hash, so lexicographic order is numeric
 * order and the boundaries are known a priori: a range at depth d is one of 2^d equal intervals.
 * Nothing is computed from the data.
 *
 * That is not only simpler than median cuts -- it removes the failure mode. With data-driven
 * boundaries a split produces two segments whose key spans are new, so they overlap other
 * segments, re-qualify for merging, split again, and the write amplification runs away (measured
 * at 200-286x). With fixed boundaries the two halves land in DISTINCT range buckets and can only
 * ever merge with same-bucket segments, so the cascade cannot form.
 *
 * A range is identified by (depth, prefix), both derived from a segment's min/max routing term by
 * integer arithmetic -- no dictionary walk, no median, no segment attribute.
 */
public class FixedRangePolicy extends MergePolicy {

    final long targetBytes;
    final int ratio;
    final String field;
    /** segment name -> {min,max} routing term; refreshed by the driver, cheap to replace with a
     *  segment attribute in a real implementation. */
    Map<String, long[]> extremes = new HashMap<>();
    /** current subdivision depth: 2^depth ranges. Grows with the index, never shrinks. */
    int depth = 0;
    int l0Trigger = 4;
    int splits, merges, idSplits, reclaims, absorbs, l0Consolidations;
    long l0ConsolidationBytes;
    /** How many documents pass through a partition merge, and how many outputs each produced.
     *  If inputDocs ~= docs indexed, every document is partitioned exactly once. If it is a
     *  multiple, "partition once" is not holding and the cause is the policy, not overhead. */
    long partitionInputDocs, partitionOutputs;
    int partitionMerges;

    /** Deleted percentage at which a range that still fills itself is rewritten in place. */
    static final int RECLAIM_PCT = Integer.getInteger("reclaimPct", 20);
    /**
     * Whether an under-full subtree is absorbed back into its parent range. On, and deliberately not
     * gated on index size even though it costs 130% more writes on a small index and buys nothing
     * there: in bytes that is 1.1 GB over the whole life of a 245 MB index, against +36% and a
     * halved query cost at 15 GB. The relative cost is worst exactly where the absolute cost is
     * negligible. An index of this kind is shared by many tenants -- that is the premise -- so a
     * small one is either young or not the workload, and a size threshold would be a knob that
     * exists only to optimise a case the design is not for.
     */
    static final boolean ABSORB = Boolean.parseBoolean(System.getProperty("absorb", "true"));
    /**
     * The dead band: a range splits at the target and is absorbed below this fraction of it.
     *
     * <p>It has to be below <b>half</b>, and that is a stability condition rather than a preference.
     * Absorbing merges a pair, so two siblings each just under the threshold produce a range of
     * twice it; at 0.5 that lands exactly on the split threshold and the merge is undone
     * immediately. Measured, the splits performed go 11 at 0.25, 19 at 0.5, 36 at 0.8 -- the churn
     * appears precisely as the threshold crosses half. A quarter leaves 2x margin.
     */
    static final double ABSORB_AT = Double.parseDouble(System.getProperty("absorbAt", "0.25"));

    /** bits of key space resolved by ONE merge: fan-out 2^FANOUT_BITS per level */
    static final int FANOUT_BITS =
            Integer.getInteger("fanoutBits", 2);   // 2^bits sub-ranges per merge
    /** Whether a range that outgrows the target descends on its own; see rule 3 in findMerges. */
    static final boolean PER_BUCKET =
            Boolean.parseBoolean(System.getProperty("perBucket", "true"));


    /** How many bits of fan-out the current index size calls for. */
    int depthTarget(SegmentInfos infos) {
        long total = 0;
        for (SegmentCommitInfo si : infos) total += sizeOf(si);
        int d = 0;
        while ((total >> d) > targetBytes && d < 12) d += FANOUT_BITS;
        return Math.max(FANOUT_BITS, d);
    }

    /** Derived from the segments themselves, in bytes -- no caller-supplied state to get wrong. */
    private void updateDepth(SegmentInfos infos) {
        long total = 0;
        for (SegmentCommitInfo si : infos) total += sizeOf(si);
        int d = 0;
        while ((total >> d) > targetBytes && d < 12) d += FANOUT_BITS;
        if (d > depth) depth = d;   // grows with the index, never shrinks
    }

    public void setDepthFor(long ignored) { /* depth is derived in findMerges now */ }

    /** True when the next findMerges would consolidate rather than split. */
    public boolean wouldConsolidate(org.apache.lucene.index.IndexWriter w) {
        return lastWasConsolidation;
    }

    boolean lastWasConsolidation;

    public FixedRangePolicy(String field, long targetBytes, int ratio) {
        this.field = field;
        this.targetBytes = targetBytes;
        this.ratio = ratio;
    }

    static int depthOf(long min, long max) {
        long diff = min ^ max;
        int d = diff == 0 ? 32 : Long.numberOfLeadingZeros(diff) - 32;
        return Math.min(d, 32);
    }

    /**
     * A segment's bucket, DERIVED rather than stored. A coarse segment written before a split spans
     * several intervals and so derives a wider bucket than its children, which is what keeps parent
     * and child from grouping together -- the property that stops merge cascades.
     *
     * <p>Two forms, and the default is the one every measurement uses. The fixed form names the
     * node of a binary tree over the key space that the segment's [min,max] falls in --
     * (depth, prefix), the bits its extremes agree on -- so two writers agree without sharing
     * anything, because there is nothing to disagree about. It requires the key space to be
     * uniform, which is why the first level of the key is hashed. The data-driven form derives the
     * narrowest interval of a stored, shared boundary set instead: it balances bytes better, at the
     * price of state. What does NOT work is per-split quantiles, where the boundaries differ per
     * segment, buckets never match and consolidation never fires.
     */
    /**
     * The two-level form: once the first level cannot separate a segment from its neighbours --
     * every document carrying one routing value -- the name continues on the second level, so a
     * whale's sub-ranges are as distinguishable as any other range. Without this they all share one
     * name, and rule 1 consolidates the halves that rule 3 has just cut.
     */
    String bucketOf(long[] e) {
        String first = bucketOf(e[0], e[1]);
        if (e[0] != e[1] || e.length < 4) {
            return first;
        }
        long diff = e[2] ^ e[3];
        int d = diff == 0 ? 64 : Long.numberOfLeadingZeros(diff);
        return first + "/" + d + ":" + Long.toHexString(d == 0 ? 0 : (e[2] >>> (64 - d)));
    }

    /**
     * Every range above this segment's own, deepest first: the candidate absorptions, in the order
     * they should be considered.
     *
     * <p>It has to be every ancestor and not just the parent. A range's data is everything in its
     * SUBTREE, so asking whether a pair of siblings is under-full by summing only what is named one
     * level down undercounts whenever either side has been split further -- and absorbing on that
     * undercount produces a range immediately over target, which splits again.
     */
    List<String> ancestorsOf(long[] e) {
        List<String> out = new ArrayList<>();
        if (e[0] == e[1] && e.length >= 4) {
            String first = bucketOf(e[0], e[1]);
            long diff = e[2] ^ e[3];
            int d2 = diff == 0 ? 64 : Long.numberOfLeadingZeros(diff);
            for (int d = d2 - 1; d >= 0; d--) {
                out.add(first + "/" + d + ":" + Long.toHexString(d == 0 ? 0 : (e[2] >>> (64 - d))));
            }
        }
        int d1 = depthOf(e[0], e[1]);
        for (int d = Math.min(d1, 32) - 1; d >= 1; d--) {
            out.add(d + ":" + Long.toHexString(e[0] >>> (32 - d)));
        }
        return out;
    }

    /** The midpoint of the second-level range currently holding [minSeq, maxSeq]. */
    static long seqMidpoint(long minSeq, long maxSeq) {
        long diff = minSeq ^ maxSeq;
        int d = diff == 0 ? 64 : Long.numberOfLeadingZeros(diff);
        if (d >= 63) return -1;                        // nothing left to cut
        long prefix = d == 0 ? 0 : (minSeq >>> (64 - d));
        return ((prefix << 1) | 1L) << (63 - d);
    }

    String bucketOf(long min, long max) {
        // the bucket is (depth, prefix), derived by arithmetic
        long diff = min ^ max;
        int d = diff == 0 ? 32 : Long.numberOfLeadingZeros(diff) - 32;
        if (d > 32) d = 32;
        long prefix = d == 0 ? 0 : (min >>> (32 - d));
        return d + ":" + Long.toHexString(prefix);
    }

    /** Midpoint of the fixed range currently holding [min,max]. */
    static long midpoint(long min, long max) {
        long diff = min ^ max;
        int depth = diff == 0 ? 32 : Long.numberOfLeadingZeros(diff) - 32;
        if (depth >= 32) return -1;               // a single hash value: needs the _id cut instead
        long prefix = depth == 0 ? 0 : (min >>> (32 - depth));
        int shift = 32 - (depth + 1);
        return ((prefix << 1) | 1L) << shift;     // first key of the upper half
    }

    static long sizeOf(SegmentCommitInfo si) {
        try { return si.sizeInBytes(); } catch (IOException e) { return Long.MAX_VALUE; }
    }

    static String hex(long v) { return String.format(Locale.ROOT, "%08x", v & 0xFFFFFFFFL); }

    @Override
    public MergeSpecification findMerges(MergeTrigger t, SegmentInfos infos, MergeContext ctx) {
        if (t != MergeTrigger.EXPLICIT) return null;
        for (SegmentCommitInfo si : infos) if (ctx.getMergingSegments().contains(si)) return null;

        updateDepth(infos);
        // The size bound before compaction: the bound is the invariant a range partition is for,
        // and compaction is an optimisation of it.
        MergeSpecification refined = refineSpec(infos);
        if (refined != null) { lastWasConsolidation = false; return refined; }
        MergeSpecification spec = new MergeSpecification();

        // 1. Size-tiered WITHIN one fixed range. Segments are grouped by (depth,prefix), which is
        //    exact -- two segments are in the same range or they are not, no overlap heuristics.
        Map<String, List<SegmentCommitInfo>> byRange = new HashMap<>();
        for (SegmentCommitInfo si : infos) {
            long[] e = extremes.get(si.info.name);
            if (e == null || sizeOf(si) > targetBytes) continue;
            // NOTE: L0 is deliberately NOT excluded here, though compacting arrived-but-
            // unpartitioned data with itself looks like pure waste -- it was measured at 18% of all
            // bytes written and two thirds of what straddles a split boundary. Excluding it was
            // tried and is a net loss: this compaction is what ACCUMULATES enough data that a
            // k-way split produces useful-sized outputs. Without it, partitioning fires on small
            // batches and 64 outputs are 64 slivers -- segments 54 -> 135, fan-out 5.0 -> 9.0 per
            // query -- which costs more than the 18% it saves. See the note's section on splitting
            // at flush: the accumulator has to come from somewhere.
            byRange.computeIfAbsent(bucketOf(e), k -> new ArrayList<>()).add(si);
        }
        for (List<SegmentCommitInfo> group : byRange.values()) {
            if (group.size() < ratio) continue;
            group.sort((a, b) -> Integer.compare(a.info.maxDoc(), b.info.maxDoc()));
            // comparable sizes only: this is what bounds write amplification to L
            List<SegmentCommitInfo> take = new ArrayList<>();
            long smallest = group.get(0).info.maxDoc();
            for (SegmentCommitInfo si : group) {
                if (take.size() >= ratio * 2) break;
                if (si.info.maxDoc() > smallest * 4L) break;
                take.add(si);
            }
            if (take.size() >= ratio) {
                spec.add(new OneMerge(take));
                merges++;
                // Consolidating L0 with itself is pure waste: these segments span the whole key
                // space, so the merge produces a bigger segment that still has to be partitioned,
                // and in the meantime it is what straddles every split boundary. Counted so the
                // cost of doing it is a number rather than an argument.
                long[] e0 = extremes.get(take.get(0).info.name);
                if (e0 != null && depthOf(e0[0], e0[1]) == 0) {
                    l0Consolidations++;
                    for (SegmentCommitInfo si : take) l0ConsolidationBytes += sizeOf(si);
                }
            }
        }
        if (!spec.merges.isEmpty()) { lastWasConsolidation = true; return spec; }

        // 1b. RECLAIM a range that has lost tenants. When a tenant leaves, its documents were
        //     concentrated in one range's segments -- that locality is the whole point of the
        //     layout -- so those segments cross the deleted-fraction threshold together and are
        //     rewritten, while under hash routing the same deletion smears across every segment and
        //     no single one ever crosses it. Measured: 3.2% of the index left deleted here against
        //     10.6% for plain tiered merging on the identical workload.
        //
        //     Note that this is what handles a LEAVING TENANT, not rule 1c below. Reclaiming in
        //     place gets the whole benefit: the 3.2% above is measured with no boundary dropped at
        //     all, and a range that is merely small costs nothing, because a range is only a name
        //     derived from the keys its segments span and an empty one is not represented at all.
        //     Absorbing a drained range on its LIVE size was measured on this workload and made
        //     things worse -- biggest range 1.15x the target to 2.90x, write amplification +28% --
        //     because such a range refills and has to be split again. That is a statement about the
        //     threshold, not about absorption: see 1c, which triggers at a quarter of the target.
        //     Note what this loop does NOT filter on: size. Rule 1 above stops consolidating a
        //     segment once it reaches the range target, so without this a segment at that ceiling
        //     would accumulate deletions with nothing left to merge it with -- the top tier is
        //     exactly where garbage would otherwise be permanent. A single-segment merge rewrites
        //     it in place, which is what TieredMergePolicy does for the same reason.
        for (SegmentCommitInfo si : infos) {
            if (si.info.maxDoc() == 0) continue;
            if (si.getDelCount() * 100L / si.info.maxDoc() < RECLAIM_PCT) continue;
            spec.add(new OneMerge(Collections.singletonList(si)));
            reclaims++;
        }
        if (!spec.merges.isEmpty()) { lastWasConsolidation = true; return spec; }

        // 1c. ABSORB an under-full subtree back into its parent range. Refinement only ever makes
        //     ranges finer, and nothing makes them coarser again, so the partition ends up bounded
        //     above and ragged below -- measured p95/median around 10x at every size above the
        //     smallest. This is the only rule that gives ground.
        //
        //     Dropping a boundary is not bookkeeping: a range's name is derived from the keys its
        //     segments span, so merging a whole subtree into one segment IS the boundary
        //     disappearing -- the output spans the parent, so it derives the parent's name.
        //
        //     The threshold sits BELOW HALF the split threshold on purpose, and that is arithmetic
        //     rather than taste: absorbing merges a PAIR, so any threshold above half produces a
        //     range at or over the split target and the merge is undone at once. Measured splits
        //     performed: 11 at 0.25x, 19 at 0.5x, 36 at 0.8x -- the churn appears exactly as the
        //     threshold crosses half. A quarter leaves 2x of margin.
        if (ABSORB) {
            Map<String, List<SegmentCommitInfo>> subtree = new HashMap<>();
            Map<String, long[]> subtreeBytes = new HashMap<>();
            Map<String, Integer> rankOf = new HashMap<>();
            for (SegmentCommitInfo si : infos) {
                long[] e = extremes.get(si.info.name);
                if (e == null) continue;
                int rank = 0;
                for (String anc : ancestorsOf(e)) {                  // deepest first
                    subtree.computeIfAbsent(anc, k -> new ArrayList<>()).add(si);
                    subtreeBytes.computeIfAbsent(anc, k -> new long[1])[0] += sizeOf(si);
                    rankOf.merge(anc, rank++, Math::min);
                }
            }
            List<String> candidates = new ArrayList<>(subtree.keySet());
            candidates.sort((a, b) -> Integer.compare(rankOf.get(a), rankOf.get(b)));
            java.util.Set<String> taken = new java.util.HashSet<>();
            for (String cand : candidates) {
                List<SegmentCommitInfo> group = subtree.get(cand);
                if (group.size() < 2) continue;                      // nothing to gain
                if (subtreeBytes.get(cand)[0] >= (long) (targetBytes * ABSORB_AT)) continue;
                boolean free = true;
                for (SegmentCommitInfo si : group) free &= taken.contains(si.info.name) == false;
                if (free == false) continue;                         // a deeper collapse has it
                for (SegmentCommitInfo si : group) taken.add(si.info.name);
                spec.add(new OneMerge(group));
                absorbs++;
            }
            if (!spec.merges.isEmpty()) { lastWasConsolidation = true; return spec; }
        }

        lastWasConsolidation = false;

        // 2. Partition ARRIVING data, in one merge, straight to the depth the index needs.
        //
        //    A document is rewritten once per level it descends, so descending level by level would
        //    cost log_k(R) rewrites. This goes straight to the target depth in ONE merge instead,
        //    so every document is written exactly once per partition event: 1<<depth outputs, and
        //    depth climbs the ladder in FANOUT_BITS steps, so k is 4, 16, 64 or 256.
        //
        //    Note which fan-out is which, since three numbers here are easy to confuse. FANOUT_BITS
        //    is the STEP OF THE DEPTH LADDER (2 bits), this rule splits arrivals k=1<<depth ways in
        //    one merge, and rule 3 -- refinement -- always descends exactly ONE bit, two ways.
        //
        //    That used to be the expensive choice and is no longer: a k-way split used to read its
        //    inputs about k+1 times, because each codec verifies every input file at the start of
        //    its merge and the whole merge ran once per output. With the inputs verified once, the
        //    postings of all outputs written from one pass, and doc values seeking to the range
        //    they own, a split reads 2.19x what it writes at k=64 -- against 2.03x for an ordinary
        //    merge, and 2.12x at k=16, so it is near enough flat in k. Fan-out stopped being a
        //    lever, which is why this partitions in one step rather than descending.
        // "Not yet partitioned" is derivable from the segment's own key span: a flush segment
        // spans the whole hash space (depth 0), whereas the outputs of a split are at depth >=
        // FANOUT_BITS by construction. No side table to keep in sync.
        List<SegmentCommitInfo> fresh = new ArrayList<>();
        for (SegmentCommitInfo si : infos) {
            long[] e = extremes.get(si.info.name);
            if (e == null) continue;
            // "Not yet partitioned" is the segment's own key span: a flush spans the whole space,
            // so depth 0, while the outputs of a split are deeper by construction.
            if (depthOf(e[0], e[1]) == 0) fresh.add(si);
        }
        if (fresh.size() >= l0Trigger && depth > 0) {
            // Partition ONCE, straight to the depth the index currently needs. A segment is never
            // re-descended afterwards: re-descending is what rewrites a document many times over,
            // and it buys nothing a coarse segment does not already give -- the segment is sorted,
            // so any slice inside it is still one contiguous docID range.
            long in = 0;
            for (SegmentCommitInfo si : fresh) in += si.info.maxDoc();
            partitionInputDocs += in;
            partitionOutputs += (1L << depth);
            partitionMerges++;
            spec.add(new FixedSplit(fresh, field, depth, 0));
            splits++;
            return spec;
        }

        return spec.merges.isEmpty() ? null : spec;
    }


    /**
     * Rule 3: a range whose own data exceeds the target descends one level, on its own.
     *
     * <p>Runs before consolidation, so that the bound on range size is never waiting on compaction
     * to finish. (Ordering was measured both ways and made almost no difference here -- the reason
     * to keep this one is that it is the invariant, not that it was faster.)
     */
    MergeSpecification refineSpec(SegmentInfos infos) {
        MergeSpecification spec = new MergeSpecification();
        // A bucket whose OWN data exceeds the target descends one level, on its own. The global depth above is derived from the TOTAL index size, so
        //    it is right for an average range and wrong for a skewed one: hashing makes buckets
        //    equal in tenant COUNT, and tenant sizes are heavy-tailed, so the bucket holding a
        //    large tenant carries many times its share. A bucket whose own data exceeds the target
        //    therefore descends one level on its own, independently of every other bucket. The
        //    tree ends up deep where the data is and shallow where it is not -- uneven in depth,
        //    even in size -- and it stays laminar, because a descent only ever ADDS the boundary
        //    that separates a node's two children.
        //
        //    The test is on the bucket's total, not on one segment: a range is over target when
        //    the data in it is, however many runs that data currently sits in.
        Map<String, List<SegmentCommitInfo>> refine = new HashMap<>();
        Map<String, Long> refineBytes = new HashMap<>();
        for (SegmentCommitInfo si : infos) {
            long[] e = extremes.get(si.info.name);
            if (e == null) continue;
            if (depthOf(e[0], e[1]) == 0) continue;          // L0, the arrival rule owns it
            String b = bucketOf(e);
            refine.computeIfAbsent(b, k -> new ArrayList<>()).add(si);
            refineBytes.merge(b, sizeOf(si), Long::sum);
        }
        for (Map.Entry<String, List<SegmentCommitInfo>> en : refine.entrySet()) {
            if (PER_BUCKET == false) break;
            if (refineBytes.get(en.getKey()) <= targetBytes) continue;
            List<SegmentCommitInfo> group = en.getValue();
            long[] e = extremes.get(group.get(0).info.name);
            if (e[0] == e[1]) {
                // One hash value holding more than a range: a whale. The hash space cannot
                // separate it any further -- every document carries the same routing value -- so
                // descend on the SECOND level of the key. The cut is the midpoint of the range's
                // own second-level interval, not the median of the data, for exactly the reason
                // the first level uses fixed subdivisions: a data-driven cut gives the two halves
                // names nobody else can derive, and consolidation stops working.
                long mid = seqMidpoint(e[2], e[3]);
                if (mid > 0) {
                    spec.add(new IdSplit(group, mid));
                    idSplits++;
                }
                continue;
            }
            int d = depthOf(e[0], e[1]);
            if (d > 0 && d < 30) {
                // One bit, so the node is replaced by its two children and nothing else moves.
                spec.add(new FixedSplit(group, field, 1, d));
                splits++;
            }
        }
        return spec.merges.isEmpty() ? null : spec;
    }

    @Override
    public MergeSpecification findForcedMerges(SegmentInfos i, int m,
            Map<SegmentCommitInfo, Boolean> s, MergeContext c) { return null; }

    @Override
    public MergeSpecification findForcedDeletesMerges(SegmentInfos i, MergeContext c) { return null; }

    /**
     * Splits one slice across two segments by cutting its docID interval in half. This is the
     * `_id` level of the key: the slice keeps one routing hash but now occupies two ranges, so it
     * can span shards and a shard boundary can fall inside it.
     */
    static class IdSplit extends OneMerge {
        private final long cut;
        IdSplit(List<SegmentCommitInfo> segs, long cut) { super(segs); this.cut = cut; }
        @Override public boolean isPartitioned() { return true; }
        @Override
        public int[][] getDocRangePartitions(List<CodecReader> readers) throws IOException {
            int[][] parts = new int[readers.size()][3];
            for (int i = 0; i < readers.size(); i++) {
                CodecReader r = readers.get(i);
                int maxDoc = r.maxDoc();
                parts[i][0] = 0;
                parts[i][2] = maxDoc;
                // The segment holds one routing value, so it is sorted by the second level alone:
                // the offset of the boundary is a binary search, not a scan.
                int lo = 0, hi = maxDoc;
                org.apache.lucene.index.NumericDocValues seq = r.getNumericDocValues("seq");
                if (seq == null) { parts[i][1] = maxDoc / 2; continue; }
                while (lo < hi) {
                    int mid = (lo + hi) >>> 1;
                    org.apache.lucene.index.NumericDocValues probe = r.getNumericDocValues("seq");
                    long v = probe.advanceExact(mid) ? probe.longValue() : Long.MAX_VALUE;
                    if (v < cut) lo = mid + 1; else hi = mid;
                }
                parts[i][1] = lo;
            }
            return parts;
        }
    }

    /**
     * Two outputs, cut at the median of the data. The index is sorted by the key, so the median
     * key is simply the value at doc maxDoc/2 -- no dictionary walk. The boundary is recorded and
     * never revisited.
     */



    /** Two outputs, cut at one given key: the shard-split operation, as a merge. */
    static class KeySplit extends OneMerge {
        final String field; final long boundary;
        KeySplit(List<SegmentCommitInfo> segs, String field, long boundary) {
            super(segs); this.field = field; this.boundary = boundary;
        }
        @Override public boolean isPartitioned() { return true; }
        @Override
        public int[][] getDocRangePartitions(List<CodecReader> readers) throws IOException {
            int[][] parts = new int[readers.size()][3];
            for (int i = 0; i < readers.size(); i++) {
                CodecReader r = readers.get(i);
                SortedDocValues dv = r.getSortedDocValues(field);
                parts[i][0] = 0;
                parts[i][2] = r.maxDoc();
                parts[i][1] = dv == null ? r.maxDoc()
                        : offsetOf(dv, firstDocPerOrd(r, dv), new BytesRef(hex(boundary)),
                                r.maxDoc());
            }
            return parts;
        }
    }

    /** 2^depth outputs at boundaries known before any data was read. */
    static class FixedSplit extends OneMerge {
        final String field; final int bits; final int fromDepth;
        FixedSplit(List<SegmentCommitInfo> segs, String field, int bits, int fromDepth) {
            super(segs); this.field = field; this.bits = bits; this.fromDepth = fromDepth;
        }
        @Override public boolean isPartitioned() { return true; }
        @Override
        public int[][] getDocRangePartitions(List<CodecReader> readers) throws IOException {
            int depth = fromDepth + bits;
            int k = 1 << bits;
            int[][] parts = new int[readers.size()][k + 1];
            for (int i = 0; i < readers.size(); i++) {
                CodecReader r = readers.get(i);
                SortedDocValues dv = r.getSortedDocValues(field);
                parts[i][0] = 0;
                parts[i][k] = r.maxDoc();
                if (dv == null) { for (int o = 1; o < k; o++) parts[i][o] = r.maxDoc(); continue; }
                int[] ordStart = firstDocPerOrd(r, dv);
                long base = 0;
                SortedDocValues probe = r.getSortedDocValues(field);
                if (probe.advanceExact(0) || probe.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                    base = Long.parseLong(probe.lookupOrd(0).utf8ToString(), 16)
                            >>> (32 - fromDepth) << (32 - fromDepth);
                }
                for (int o = 1; o < k; o++) {
                    long bound = base + (((long) o) << (32 - depth));
                    parts[i][o] = offsetOf(dv, ordStart, new BytesRef(hex(bound)), r.maxDoc());
                }
                for (int o = 1; o <= k; o++) {
                    if (parts[i][o] < parts[i][o - 1]) parts[i][o] = parts[i][o - 1];
                }
            }
            return parts;
        }
    }

    static int[] firstDocPerOrd(CodecReader r, SortedDocValues dv) throws IOException {
        int n = dv.getValueCount();
        int[] s = new int[n + 1];
        java.util.Arrays.fill(s, -1);
        SortedDocValues scan = r.getSortedDocValues("routing");
        for (int d = scan.nextDoc(); d != DocIdSetIterator.NO_MORE_DOCS; d = scan.nextDoc()) {
            if (s[scan.ordValue()] == -1) s[scan.ordValue()] = d;
        }
        s[n] = r.maxDoc();
        for (int i = n - 1; i >= 0; i--) if (s[i] == -1) s[i] = s[i + 1];
        return s;
    }

    static int offsetOf(SortedDocValues dv, int[] ordStart, BytesRef key, int maxDoc)
            throws IOException {
        int ord = dv.lookupTerm(key);
        if (ord < 0) ord = -ord - 1;
        if (ord >= ordStart.length - 1) return maxDoc;
        return ordStart[ord];
    }

}
