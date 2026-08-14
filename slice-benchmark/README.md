# The benchmark behind the numbers

Every figure quoted for the range-partitioned merge comes from here. It is plain Java against
`lucene-core` with no build integration on purpose — it has to be runnable against a jar built from
this branch, which is the thing being measured.

```sh
./gradlew :lucene:core:jar
CORE=lucene/core/build/libs/lucene-core-11.0.0-SNAPSHOT.jar
javac -cp $CORE -d /tmp/sb slice-benchmark/*.java
java -cp /tmp/sb:$CORE -DnoCFS=true -Dscale=4 -DindexMb=300 ContainerIndependence
```

Three arms over one corpus — `nomerge` (the denominator), `stock` (`TieredMergePolicy`), `ranged`
(`FixedRangePolicy`) — reporting bytes and segments per single-tenant query, write amplification
against a no-merge build, and merge IO attributed **per `OneMerge` and per file extension**.

## The configurations the numbers come from

| what | flags |
|---|---|
| headline | `-DnoCFS=true -Dscale=4 -DindexMb=300` |
| text corpus, where the terms dictionary dominates | `-Dtext=true -Dscale=1 -DindexMb=122` |
| with a points field | `-Dpoints=true` |
| tenants leaving, one whale deleted | `-Dchurn=25 -DdeleteWhale=2` |
| four times finer ranges | `-DtargetMult=0.25` |

`-DindexMb` must match the corpus the other flags actually build: it sets the segment cap by
dividing, and getting it wrong silently measures a different regime — at the wrong value `stock`
performs no merges at all and every comparison against it is void.

## Things that were measured wrong here before, and are worth not repeating

- **Attributing merge IO from a per-round prediction.** `maybeMerge()` drains the policy in a loop
  and runs several rules per round, so a per-round guess puts splits and consolidations in one
  bucket. It produced a conclusion that was exactly backwards. Attribution is now per `OneMerge`,
  through a custom `MergeScheduler`.
- **A stored payload of 400 identical characters**, which compressed to nothing and made stored
  fields a rounding error. Real `_source` is varied, and it turns out to be the largest thing in the
  index.
- **Two arms through different query paths** — one a raw `TermsEnum` seek, the other a full search.
  Both arms now run one path and retrieve postings rather than short-circuiting on `docFreq`.
- **Probing a tenant that was never indexed**, which measured the cost of a miss.
