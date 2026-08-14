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

## Global value order: solved

`TestPerRangePointsFormat` is green (2/2, seeds 1/2/3/11/42, and the whole `:lucene:sandbox:test`
suite passes). It took two fixes, both of them contracts rather than slips:

1. **Ordering.** `CheckIndex` requires a full traversal of a ONE-data-dimension field to sweep the
   values once, ascending, tie-broken by increasing docID. Traversal order is the tree SHAPE --
   `PointValues.intersect` descends with `moveToChild`/`moveToSibling` -- so a synthetic root whose
   children are ranges is range-major by construction, and the ranges each span the whole value
   space, so no arrangement of them helps. The interleaving has to be point by point: a flat tree
   whose `visitDocValues` runs a k-way merge over one cursor per range. Multi-dimension fields have
   NO such contract (leaves are ordered by whichever dimension compresses best), so they keep the
   hierarchical tree and its better pruning.

2. **Cell announcement.** `CheckIndex` also requires every visited value to lie inside the cell of
   the most recent `compare()` call. Cursors call `compare` to prune while hunting their next leaf,
   so the merge would emit a point from range A right after range B had asked about a cell nowhere
   near it. Each cursor now captures the bounds of the leaf it buffered, and the merge re-announces
   them whenever it starts drawing from a different leaf.

Bug found along the way, worth remembering: `fillNextLeaf` can buffer a leaf AND reach the end of
the tree in the same call, so `next()` must check the buffer BEFORE the exhausted flag. It did not,
so every cursor yielded exactly one point -- a full traversal saw 8 points instead of 200, and
queries returned 0.

Costs, honestly: a one-dimension whole-index traversal now pays a heap operation per point, cannot
use the `visitDocIDs` shortcut for cells lying entirely inside the query (the merge needs values to
order by), and may re-announce a cell per point when ranges interleave tightly. All of it falls on
whole-index queries only -- a slice-scoped query reads one range and gets an ordinary block k-d tree
with nothing added, which is the case the format exists for.

## Still unmeasured

The claim that per-range points make a partitioned merge cost 1x instead of kx. The benchmark corpus
(`.agents/slice-sim/ContainerIndependence.java`) has no points field at all, so the 12.57 merge read
amplification measured for postings understates a real ES index. Adding an IntPoint/LongPoint field
to `doc()` is the next step.
