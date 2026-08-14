# Range-partitioned merge — what is on this branch

A merge that writes several output segments, each holding a contiguous doc range of every input.
Built for multi-tenant ("slice") indices, where an index sorted by `(h(routing), routing, _id)` makes
a doc range a key range — but nothing in the API mentions a tenant, and the same primitive serves
resharding, time-tiering, or dropping a key range.

The design note this serves is *Range routing*
(<https://claude.ai/code/artifact/340e59b5-daac-42bd-9f09-59b9931da61b>); this file is only the
Lucene-side status, so that the code and the note do not drift.

## The pieces, and whether each one earns its place

Measured with `.agents/slice-sim/ContainerIndependence.java` in the Elasticsearch worktree, against
plain `TieredMergePolicy` on the same corpus.

| piece | ~lines | load-bearing? | evidence |
|---|---|---|---|
| the primitive — `OneMerge.isPartitioned`/`getDocRangePartitions`, `DocRangeCodecReader`, `IndexWriter.multiOutputMergeMiddle`, `SegmentInfos.applyMergeChanges` | 1,000 | it **is** the feature | 2,630 bytes and 6.0 segments per single-slice query, against 10,490 and 26.0 |
| single-pass postings — `TermsPushWriter`, `MultiOutputTermsMerger`, `PerFieldPostingsFormat` dispatch | 500 | **yes** | A/B on a text corpus at k=64: a split merge reads 2.62x its writes with it, **9.59x without**; `.tim` 58 -> 682 MB |
| doc-values narrowing — `DocRangeDocValuesProducer` | 256 | **yes** | `dvd` 15.35x -> 8.50x |
| verify inputs once — `OneMerge.areInputsVerified`, early return in `CodecUtil` | 60 | **yes** | split merge read/write 14.03 -> 2.19 |
| `TermsEnum.docFreq(minDoc, maxDoc)` | 39 | independent (per-slice BM25) | — |
| per-range points — `lucene/sandbox/.../perrangepoints` | 1,200 | **no, at this ratio** | recovers 73 MB of 1,124, i.e. 6.5% of merge reads |

**The headline:** a partitioned merge reads **2.19x** what it writes, against **2.03x** for an
ordinary merge. It is not proportional to the output count. Every Lucene merge reads its inputs twice
— once to verify, once to copy — and this one now costs the same.

## What each fix was for

1. **The checksum, not the dictionary.** Every codec opens its merge by checksumming each input file
   end to end. A partitioned merge runs a whole merge per output, so that repeated k times: reads
   ~= `(k+1)` x inputs to write them once. Verify once per merge instead.
2. **The postings.** A terms dictionary is ordered by term, so a term's postings are spread across
   every output and masking documents saves nothing — unlike doc values or stored fields, where an
   output's documents are a contiguous interval a reader seeks past. One walk of the inputs feeds
   every output's writer.
3. **Doc values.** A merge decodes every value and drops what maps to -1, so each output read the
   whole column. The iterators now seek to the output's range and stop at its end.

Left over, and understood: stored fields are chunked, so each output re-reads the chunks straddling
its boundaries (2.08 at k=64, 2.75 at k=256 — it only bites once an output's share of one input
approaches a chunk). Points are read once per output because a block k-d tree is ordered by value and
no leaf can be skipped by doc range; that is what the sandbox format fixes, and why it is parked
rather than deleted.

## Preconditions, deliberately fatal

A partitioned merge refuses rather than falling back, because the fallback is the cost partitioning
exists to avoid and taking it silently would leave no sign:

* **an index sort** — without it `DocIDMerger` concatenates input by input, so an output's documents
  are not one contiguous run of the merged space and splitting postings by doc id would hand
  documents to the wrong output;
* **a postings format implementing the push path** — today only the default block-tree does, so
  `TestMultiOutputMerge` must pin `TestUtil.getDefaultCodec()` rather than take a randomised one.

## Upstream order, easiest first

1. `TermsEnum.docFreq(minDoc, maxDoc)` — 39 lines and a test, no dependencies.
2. The partitioned merge primitive. It is now a *generalisation* of the existing merge path rather
   than a copy of it: packaging a finished segment and committing a merge's outputs are one method
   each, and an ordinary merge is the case where the list of outputs has one entry.
3. Verify once per merge. Small, and a win for every merge on object storage.
4. The push API — the contentious one, since it asks postings formats to grow a second write path.

## Known gaps

* **Points merge is still k-fold** unless the sandbox format is used, which is not wired into the
  benchmark. Measured cost above.
* **Norms are not narrowed.** `NormsConsumer.mergeNormsField` walks the column with `nextDoc()`,
  exactly the shape the doc-values narrowing fixed, so each output reads the whole norms column. The
  benchmark corpus keeps norms under a megabyte, so this is unmeasured rather than dismissed.
* **KNN vectors are not narrowed** either, and would be read once per output.
