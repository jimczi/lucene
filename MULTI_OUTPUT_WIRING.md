# Wiring the single-pass postings merge into IndexWriter

State: `TermsPushWriter`, `MultiOutputTermsMerger` and the `PerFieldPostingsFormat` dispatch are
committed and tested. Nothing calls `MultiOutputTermsMerger` yet, so `multiOutputMergeMiddle` still
runs a whole merge per output and the measured cost stands unchanged.

## The constraint that shapes it

`PerFieldPostingsFormat` stamps `PER_FIELD_FORMAT_KEY` and `PER_FIELD_SUFFIX_KEY` onto each
`FieldInfo` while postings are written (`SegmentMerger.merge()` line ~149), and those attributes are
persisted by the field-infos phase (line ~175, last). Move the postings pass after `merge()` returns
and the attributes are written before they exist: every output segment becomes unreadable, and it
fails at open time rather than at merge time.

So the shared pass must run *between* those two phases, for all outputs at once. `merge()` has to be
split -- a skip flag is not enough.

## Shape

1. `SegmentMerger.mergeUpToPostings()` -- everything through norms, returning the
   `SegmentWriteState`/`SegmentReadState` it built.
2. Caller runs either `MultiOutputTermsMerger.merge(...)` once for all outputs, or the existing
   per-output `mergeTerms` when `supportsPushWriter()` is false.
3. `SegmentMerger.mergeAfterPostings()` -- doc values, points, vectors, term vectors, field infos.
4. `multiOutputMergeMiddle` splits its single loop into: phase A over all outputs (step 1), the
   shared pass (step 2), phase B over all outputs (step 3 plus the existing `setFiles`,
   compound-file and abort handling).

## The merged doc space

`MultiOutputTermsMerger` needs a `MergeState` over the **unmasked** readers, plus `outputStarts`:
`k+1` boundaries in merged doc space. Each output's own `MergeState` is masked by
`DocRangeCodecReader` and cannot supply this.

`outputStarts[o+1] - outputStarts[o]` is the live-doc count of output `o`, which each phase-A
`SegmentMerger` already knows (`segmentInfo.maxDoc()`), so the boundaries are a prefix sum -- no
counting pass needed.

Correctness rests on outputs being contiguous *and in order* in the merged doc space. That holds
under an index sort, where merged order is key order and outputs are key ranges. It does **not**
hold otherwise: without a sort, `DocIDMerger` concatenates input by input, so each output's docs
land in one block per input rather than one run. Gate the single-pass path on the merge having an
index sort, and assert the boundaries are non-decreasing.

## Verification

- `TestMultiOutputMerge` covers it once wired; add a case asserting the single-pass and per-output
  paths produce byte-identical postings for the same input.
- A codec whose format reports `supportsPushWriter() == false` must still merge correctly via the
  fallback -- worth an explicit test, since that path is otherwise never taken in-tree.
- Benchmark: `.agents/slice-sim/ContainerIndependence.java`, `-Dtext=true -Dscale=4
  -DindexMb=110 -DfanoutBits=3`. Merge read amplification is 70.03 today; the prediction is roughly
  8.8, i.e. near stock's 5.60, with the read win (4,612 bytes/q against stock 18,152) unchanged.
