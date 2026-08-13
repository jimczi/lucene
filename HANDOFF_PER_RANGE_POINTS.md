# Per-range points: state and the open problem

Branch `agent/per-range-points`, off `agent/multi-output-wiring`.

## Why points are stored per range and terms are not

A terms dictionary shares keys: one entry for a term serves every range, so splitting it writes
the term bytes and block-tree index once per range. Measured: dict x15.66 at R=64. A block k-d tree
shares nothing -- every (value, doc) pair belongs to one document -- so splitting only redistributes
pairs. Leaves hold 512 points, so split values are a small fraction, plus one partly filled leaf per
range.

Sharing also hurts: one tree is value-ordered, so every leaf interleaves all ranges' documents, and
a range query for a small slice traverses leaves full of other slices' documents. It costs what the
segment costs, not what the slice costs.

And it is what makes a partitioned merge affordable. An output keeps its own range and maps the rest
to -1; with one tree it must read all the points to discover that, so k outputs read them k times.

## Built

- `PerRangePointsFormat(delegate, DocRanges)` -- one sub-index per doc range, suffix `<outer>R<n>`,
  plus a `.prpm` metadata file holding the boundaries and which ranges have a sub-index.
- `PerRangePointsWriter` -- `writeField` splits by range through a filtering `PointTree`;
  `merge(MergeState)` narrows each input to its surviving ranges via
  `PerRangePointsReader.survivingOnly(docMap)` and substitutes them into a `MergeState`, then
  defers to the inherited merge. Points have no per-field dispatching format, so recognising our
  own reader is sound rather than a guess about wrappers -- unlike the postings case.
- `PerRangePointsReader` + `UnionPointValues` -- the ranges as children of a synthetic root.
- `PerRangePointsTestCodec` registered via `src/test/META-INF/services`. Needed because a segment
  records only a codec NAME: a FilterCodec reusing the base name is read back with the stock points
  format. That cost one debugging round.

## The open problem: global value order

`TestPerRangePointsFormat` fails in `CheckIndex.VerifyPointsVisitor`, not on results:

    packed points value ... is out-of-order vs the previous document's value

`CheckIndex` requires a full `PointValues` traversal to yield values in GLOBAL order. The union
walks range 0's tree, then range 1's -- ordered within a range, not across them.

This does not affect query correctness (`PointRangeQuery` collects matches and does not care about
order) and it does not affect a slice-scoped query at all, which reads one range and is perfectly
ordered. It is confined to whole-index traversals. Options, in the order worth trying:

1. Make `UnionPointValues` a k-way merge in value order. Correct, and costs a heap over R cursors
   on every full traversal -- exactly the case per-range points is worst at anyway.
2. Decide the ordering contract applies per sub-tree and relax `CheckIndex`. Cheaper, but it is a
   Lucene contract change and needs its own argument.
3. Expose the ranges rather than a union, so a whole-index traversal is the caller's loop over
   ranges and no single `PointValues` claims global order.

Nothing here is measured yet. The claim that per-range points make a partitioned merge cost 1x
instead of kx is still a prediction; the benchmark corpus has no points field, so it has never
exercised this at all.
