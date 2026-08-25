/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.lucene.index;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.store.MockDirectoryWrapper;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.TestUtil;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.BytesRef;

/** Tests for merges that produce several output segments. */
public class TestMultiOutputMerge extends LuceneTestCase {

  private static final int OUTPUTS = 3;
  private static final int SEGMENTS = 4;
  private static final int PER_SEGMENT = 120;
  private static final String SOFT = "soft_deleted";

  private CountDownLatch mergeStarted;
  private CountDownLatch proceed;
  private volatile boolean enabled;

  @Override
  public void setUp() throws Exception {
    super.setUp();
    mergeStarted = new CountDownLatch(1);
    proceed = new CountDownLatch(1);
    enabled = false;
  }

  private IndexWriterConfig config() {
    IndexWriterConfig iwc = newIndexWriterConfig(null);
    // A contiguous doc range is a contiguous key range only when sorted.
    iwc.setIndexSort(new Sort(new SortField("sort", SortField.Type.STRING)));
    iwc.setMergePolicy(new PartitioningMergePolicy());
    return iwc;
  }

  private static Document doc(String id, long val) {
    Document d = new Document();
    d.add(new StringField("id", id, Field.Store.NO));
    d.add(new SortedDocValuesField("sort", new BytesRef(id)));
    d.add(new StoredField("id", id));
    d.add(new NumericDocValuesField("val", val));
    return d;
  }

  /** A document whose vector is derived from its id, so a mismatch after a merge is detectable. */
  private static Document vectorDoc(String id, float[] vector) {
    Document d = doc(id, 0);
    d.add(new KnnFloatVectorField("vec", vector, VectorSimilarityFunction.EUCLIDEAN));
    return d;
  }

  private static float[] vectorFor(int seg, int d) {
    return new float[] {seg, d, (float) (seg * PER_SEGMENT + d)};
  }

  private static String id(int seg, int d) {
    return String.format(java.util.Locale.ROOT, "id-%02d-%04d", seg, d);
  }

  /** Every live document's stored id. */
  private static List<String> liveIds(DirectoryReader r) throws IOException {
    List<String> out = new ArrayList<>();
    for (LeafReaderContext ctx : r.leaves()) {
      StoredFields sf = ctx.reader().storedFields();
      Bits live = ctx.reader().getLiveDocs();
      for (int d = 0; d < ctx.reader().maxDoc(); d++) {
        if (live == null || live.get(d)) {
          out.add(sf.document(d).get("id"));
        }
      }
    }
    return out;
  }

  private static void assertEachOutputSorted(DirectoryReader r) throws IOException {
    for (LeafReaderContext ctx : r.leaves()) {
      StoredFields sf = ctx.reader().storedFields();
      String prev = null;
      for (int d = 0; d < ctx.reader().maxDoc(); d++) {
        String v = sf.document(d).get("id");
        if (prev != null) {
          assertTrue("output not index-sorted: " + prev + " then " + v, v.compareTo(prev) >= 0);
        }
        prev = v;
      }
    }
  }

  /** A merge producing several outputs keeps every document exactly once. */
  public void testProducesMultipleOutputs() throws Exception {
    try (Directory dir = newDirectory()) {
      Set<String> expected = new HashSet<>();
      try (IndexWriter w = new IndexWriter(dir, config())) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int d = 0; d < PER_SEGMENT; d++) {
            w.addDocument(doc(id(seg, d), 0));
            expected.add(id(seg, d));
          }
          w.flush();
        }
        w.commit();
        enabled = true;
        proceed.countDown();
        w.maybeMerge();
      }
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        assertTrue("expected several outputs, got " + r.leaves().size(), r.leaves().size() > 1);
        List<String> live = liveIds(r);
        assertEquals(expected.size(), live.size());
        assertEquals(expected, new HashSet<>(live));
        assertEachOutputSorted(r);
      }
    }
  }

  /** Deletes arriving after the merge snapshot must land on the output owning the doc. */
  public void testConcurrentDeletes() throws Exception {
    try (Directory dir = newDirectory()) {
      Set<String> expected = new HashSet<>();
      List<String> deleted = new ArrayList<>();
      try (IndexWriter w = new IndexWriter(dir, config())) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int d = 0; d < PER_SEGMENT; d++) {
            w.addDocument(doc(id(seg, d), 0));
            expected.add(id(seg, d));
          }
          w.flush();
        }
        w.commit();
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int d = 5; d < PER_SEGMENT; d += 17) {
            deleted.add(id(seg, d));
          }
        }
        expected.removeAll(deleted);

        Thread deleter =
            new Thread(
                () -> {
                  try {
                    mergeStarted.await();
                    for (String id : deleted) {
                      w.deleteDocuments(new Term("id", id));
                    }
                    // Force the buffered deletes to resolve against the merging
                    // segments so they must be carried over.
                    DirectoryReader.open(w).close();
                  } catch (Throwable t) {
                    throw new AssertionError(t);
                  } finally {
                    proceed.countDown();
                  }
                });
        deleter.start();
        enabled = true;
        w.maybeMerge();
        deleter.join();
      }
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        List<String> live = liveIds(r);
        assertEquals("no document may be duplicated", live.size(), new HashSet<>(live).size());
        assertEquals(expected, new HashSet<>(live));
        assertEachOutputSorted(r);
      }
    }
  }

  /** Doc-values updates arriving mid-merge must be remapped to the owning output. */
  public void testConcurrentDocValuesUpdates() throws Exception {
    try (Directory dir = newDirectory()) {
      Map<String, Long> expected = new HashMap<>();
      try (IndexWriter w = new IndexWriter(dir, config())) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int d = 0; d < PER_SEGMENT; d++) {
            w.addDocument(doc(id(seg, d), 1));
            expected.put(id(seg, d), 1L);
          }
          w.flush();
        }
        w.commit();

        List<String> updates = new ArrayList<>();
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int d = 7; d < PER_SEGMENT; d += 19) {
            updates.add(id(seg, d));
          }
        }
        updates.forEach(id -> expected.put(id, 42L));

        Thread updater =
            new Thread(
                () -> {
                  try {
                    mergeStarted.await();
                    for (String id : updates) {
                      w.updateNumericDocValue(new Term("id", id), "val", 42L);
                    }
                    DirectoryReader.open(w).close();
                  } catch (Throwable t) {
                    throw new AssertionError(t);
                  } finally {
                    proceed.countDown();
                  }
                });
        updater.start();
        enabled = true;
        w.maybeMerge();
        updater.join();
      }
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        for (LeafReaderContext ctx : r.leaves()) {
          StoredFields sf = ctx.reader().storedFields();
          NumericDocValues dv = ctx.reader().getNumericDocValues("val");
          Bits live = ctx.reader().getLiveDocs();
          for (int d = 0; d < ctx.reader().maxDoc(); d++) {
            if (live != null && live.get(d) == false) {
              continue;
            }
            String id = sf.document(d).get("id");
            assertNotNull(dv);
            assertTrue("no value for " + id, dv.advanceExact(d));
            assertEquals("wrong value for " + id, (long) expected.get(id), dv.longValue());
          }
        }
      }
    }
  }

  /** Soft deletes must be accounted per output, not for the merge as a whole. */
  public void testSoftDeletes() throws Exception {
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = config();
      iwc.setSoftDeletesField(SOFT);
      iwc.setMergePolicy(
          new SoftDeletesRetentionMergePolicy(
              SOFT, MatchAllDocsQuery::new, new PartitioningMergePolicy()));
      Set<String> originals = new HashSet<>();
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int d = 0; d < PER_SEGMENT; d++) {
            w.addDocument(doc(id(seg, d), 0));
            originals.add(id(seg, d));
          }
          w.flush();
        }
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int d = 3; d < PER_SEGMENT; d += 23) {
            Document tomb = doc(id(seg, d), 0);
            tomb.add(new NumericDocValuesField(SOFT, 1));
            w.softUpdateDocument(
                new Term("id", id(seg, d)), tomb, new NumericDocValuesField(SOFT, 1));
          }
        }
        w.commit();
        enabled = true;
        proceed.countDown();
        w.maybeMerge();
      }
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        assertTrue(r.leaves().size() > 1);
        // Retention keeps tombstones, so ids may repeat; nothing may be lost.
        assertTrue(new HashSet<>(liveIds(r)).containsAll(originals));
      }
    }
  }

  /** A malformed partition spec must be rejected, not silently corrupt the index. */
  public void testRejectsBadPartitions() throws Exception {
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = config();
      iwc.setMergePolicy(new BadPartitioningMergePolicy());
      // Serial, so the validation failure surfaces on this thread rather than
      // on a merge thread where the assertion could not observe it.
      iwc.setMergeScheduler(new SerialMergeScheduler());
      Throwable caught = null;
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int seg = 0; seg < 2; seg++) {
          for (int d = 0; d < 20; d++) {
            w.addDocument(doc(id(seg, d), 0));
          }
          w.flush();
        }
        w.commit();
        enabled = true;
        proceed.countDown();
        try {
          w.maybeMerge();
        } catch (Throwable t) {
          caught = t;
        }
        w.rollback();
      } catch (Throwable t) {
        if (caught == null) {
          caught = t;
        }
      }
      assertNotNull("a malformed partition spec must be rejected", caught);
      boolean sawIae = false;
      for (Throwable t = caught; t != null; t = t.getCause()) {
        if (t instanceof IllegalArgumentException) {
          sawIae = true;
          break;
        }
      }
      assertTrue("expected IllegalArgumentException in the cause chain, got " + caught, sawIae);
    }
  }

  /**
   * An IOException while writing one of the outputs must leave the index exactly as it was: a
   * partitioned merge is all-or-nothing, like any other merge.
   */
  public void testIOExceptionWritingAnOutput() throws Exception {
    try (Directory dir = newDirectory()) {
      Set<String> committed = new HashSet<>();
      IndexWriterConfig iwc = config();
      iwc.setMergeScheduler(new SerialMergeScheduler());
      IndexWriter w = new IndexWriter(dir, iwc);
      for (int seg = 0; seg < SEGMENTS; seg++) {
        for (int d = 0; d < PER_SEGMENT; d++) {
          w.addDocument(doc(id(seg, d), 0));
          committed.add(id(seg, d));
        }
        w.flush();
      }
      w.commit();

      AtomicBoolean fired = new AtomicBoolean();
      if (dir instanceof MockDirectoryWrapper mock) {
        // Fail once, partway through writing the partitioned outputs.
        mock.failOn(
            new MockDirectoryWrapper.Failure() {
              @Override
              public void eval(MockDirectoryWrapper d) throws IOException {
                if (fired.get()) {
                  return;
                }
                for (StackTraceElement e : Thread.currentThread().getStackTrace()) {
                  if ("multiOutputMergeMiddle".equals(e.getMethodName())) {
                    fired.set(true);
                    throw new IOException("injected while writing a partitioned output");
                  }
                }
              }
            });
      }

      enabled = true;
      proceed.countDown();
      try {
        w.maybeMerge();
      } catch (Throwable expected) {
        // the injected failure may surface here
      }
      if (dir instanceof MockDirectoryWrapper) {
        assertTrue("the failure must actually have been injected", fired.get());
      }
      // The injected Failure disables itself after firing once.
      try {
        w.rollback();
      } catch (Throwable ignored) {
        // writer may already be tragically closed by the injected failure
      }

      // Whatever happened, the committed index must still be intact and valid.
      TestUtil.checkIndex(dir);
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        List<String> live = liveIds(r);
        assertEquals("no document may be duplicated", live.size(), new HashSet<>(live).size());
        assertEquals(committed, new HashSet<>(live));
      }
    }
  }

  /** rollback() while a partitioned merge is in flight must revert to the last commit. */
  public void testRollbackDuringPartitionedMerge() throws Exception {
    try (Directory dir = newDirectory()) {
      Set<String> committed = new HashSet<>();
      IndexWriter w = new IndexWriter(dir, config());
      for (int seg = 0; seg < SEGMENTS; seg++) {
        for (int d = 0; d < PER_SEGMENT; d++) {
          w.addDocument(doc(id(seg, d), 0));
          committed.add(id(seg, d));
        }
        w.flush();
      }
      w.commit();

      Thread roller =
          new Thread(
              () -> {
                try {
                  mergeStarted.await();
                  // Release the merge first: rollback() waits for in-flight
                  // merges, so holding it parked here would deadlock.
                  proceed.countDown();
                  w.rollback();
                } catch (Throwable ignored) {
                  // rollback races the merge; either ordering is acceptable
                }
              });
      roller.start();
      enabled = true;
      try {
        w.maybeMerge();
      } catch (Throwable ignored) {
        // may surface the abort
      }
      roller.join();
      try {
        w.close();
      } catch (Throwable ignored) {
      }

      TestUtil.checkIndex(dir);
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        List<String> live = liveIds(r);
        assertEquals("no document may be duplicated", live.size(), new HashSet<>(live).size());
        assertEquals("rollback must restore the committed state", committed, new HashSet<>(live));
      }
    }
  }

  /** A wrapper that drops the partitioning must fail loudly, not silently make one segment. */
  public void testWrappingMustPreservePartitioning() throws Exception {
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = config();
      iwc.setMergeScheduler(new SerialMergeScheduler());
      iwc.setMergePolicy(
          new OneMergeWrappingMergePolicy(
              new PartitioningMergePolicy(),
              toWrap ->
                  // Deliberately forgets to forward isPartitioned().
                  new MergePolicy.OneMerge(toWrap.segments) {
                    @Override
                    public CodecReader wrapForMerge(CodecReader reader) throws IOException {
                      return toWrap.wrapForMerge(reader);
                    }
                  }));
      Throwable caught = null;
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int seg = 0; seg < 2; seg++) {
          for (int d = 0; d < 20; d++) {
            w.addDocument(doc(id(seg, d), 0));
          }
          w.flush();
        }
        w.commit();
        enabled = true;
        proceed.countDown();
        try {
          w.maybeMerge();
        } catch (Throwable t) {
          caught = t;
        }
        w.rollback();
      } catch (Throwable t) {
        if (caught == null) {
          caught = t;
        }
      }
      assertNotNull("dropping the partitioning must be reported", caught);
      boolean sawIse = false;
      for (Throwable t = caught; t != null; t = t.getCause()) {
        if (t instanceof IllegalStateException
            && t.getMessage() != null
            && t.getMessage().contains("dropped the partitioning")) {
          sawIse = true;
          break;
        }
      }
      assertTrue("expected the wrapping check to fire, got " + caught, sawIse);
    }
  }

  /** FilterOneMerge forwards partitioning, so wrapping through it still produces k outputs. */
  public void testFilterOneMergePreservesPartitioning() throws Exception {
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = config();
      iwc.setMergePolicy(
          new OneMergeWrappingMergePolicy(
              new PartitioningMergePolicy(), MergePolicy.FilterOneMerge::new));
      Set<String> expected = new HashSet<>();
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int d = 0; d < PER_SEGMENT; d++) {
            w.addDocument(doc(id(seg, d), 0));
            expected.add(id(seg, d));
          }
          w.flush();
        }
        w.commit();
        enabled = true;
        proceed.countDown();
        w.maybeMerge();
      }
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        assertTrue(
            "wrapping through FilterOneMerge must keep several outputs", r.leaves().size() > 1);
        assertEquals(expected, new HashSet<>(liveIds(r)));
      }
    }
  }

  // ------------------------------------------------------------------

  /**
   * Vector fields survive a partitioned merge without the KNN format knowing about partitioning.
   *
   * <p>Nothing narrows vector values by document range, so each output reads every input's vectors
   * and keeps the ones its range owns. Its graph is then built from scratch: a graph is reused only
   * when a reader's deleted fraction is small, and an output of a k-way split presents everything
   * outside its own range as deleted, which for any k &gt; 1 is at least half the segment.
   */
  public void testVectorsSurvivePartitionedMerge() throws Exception {
    try (Directory dir = newDirectory()) {
      Map<String, float[]> expected = new HashMap<>();
      try (IndexWriter w = new IndexWriter(dir, config())) {
        for (int seg = 0; seg < SEGMENTS; seg++) {
          for (int d = 0; d < PER_SEGMENT; d++) {
            float[] vector = vectorFor(seg, d);
            w.addDocument(vectorDoc(id(seg, d), vector));
            expected.put(id(seg, d), vector);
          }
          w.flush();
        }
        w.commit();
        enabled = true;
        proceed.countDown();
        w.maybeMerge();
      }

      try (DirectoryReader r = DirectoryReader.open(dir)) {
        assertTrue("expected several outputs, got " + r.leaves().size(), r.leaves().size() > 1);

        // Every vector is present exactly once, still attached to its own document.
        Map<String, float[]> found = new HashMap<>();
        for (LeafReaderContext ctx : r.leaves()) {
          StoredFields sf = ctx.reader().storedFields();
          FloatVectorValues values = ctx.reader().getFloatVectorValues("vec");
          assertNotNull("output lost the vector field entirely", values);
          KnnVectorValues.DocIndexIterator it = values.iterator();
          for (int doc = it.nextDoc();
              doc != KnnVectorValues.DocIndexIterator.NO_MORE_DOCS;
              doc = it.nextDoc()) {
            String id = sf.document(doc).get("id");
            assertNull("vector for " + id + " appeared in two outputs", found.put(id, null));
            assertArrayEquals(
                "vector changed for " + id, expected.get(id), values.vectorValue(it.index()), 0f);
          }
        }
        assertEquals(expected.keySet(), found.keySet());

        // And the rebuilt graphs are searchable: a query at a known vector finds its own document.
        IndexSearcher searcher = new IndexSearcher(r);
        float[] target = vectorFor(SEGMENTS - 1, PER_SEGMENT - 1);
        TopDocs hits = searcher.search(new KnnFloatVectorQuery("vec", target, 1), 1);
        assertEquals(1, hits.scoreDocs.length);
        assertEquals(
            id(SEGMENTS - 1, PER_SEGMENT - 1),
            searcher.storedFields().document(hits.scoreDocs[0].doc).get("id"));
      }
    }
  }

  private class PartitioningMergePolicy extends MergePolicy {
    @Override
    public MergeSpecification findMerges(MergeTrigger t, SegmentInfos infos, MergeContext ctx) {
      if (enabled == false || infos.size() < 2) {
        return null;
      }
      List<SegmentCommitInfo> segs = new ArrayList<>();
      for (SegmentCommitInfo si : infos) {
        if (ctx.getMergingSegments().contains(si)) {
          return null;
        }
        segs.add(si);
      }
      int[][] parts = new int[segs.size()][];
      for (int i = 0; i < segs.size(); i++) {
        int maxDoc = segs.get(i).info.maxDoc();
        int[] b = new int[OUTPUTS + 1];
        for (int o = 0; o <= OUTPUTS; o++) {
          b[o] = (int) ((long) o * maxDoc / OUTPUTS);
        }
        parts[i] = b;
      }
      MergeSpecification spec = new MergeSpecification();
      spec.add(new Partitioned(segs, parts));
      // One partitioned merge per test, and then done. This policy has no fixed point -- it
      // answers every request while there is more than one segment, and it turns OUTPUTS
      // segments back into OUTPUTS segments -- while IndexWriter asks the policy again after
      // every merge it finishes. Under the concurrent scheduler the writer closes out from
      // under that; under the serial one, which the framework picks at random, the scheduler
      // drains the queue in a loop and the merges never stop.
      enabled = false;
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

  /** Emits boundaries that do not cover every document. */
  private class BadPartitioningMergePolicy extends PartitioningMergePolicy {
    @Override
    public MergeSpecification findMerges(MergeTrigger t, SegmentInfos infos, MergeContext ctx) {
      MergeSpecification spec = super.findMerges(t, infos, ctx);
      if (spec == null) {
        return null;
      }
      MergeSpecification bad = new MergeSpecification();
      for (OneMerge m : spec.merges) {
        int[][] parts = ((Partitioned) m).parts;
        parts[0][parts[0].length - 1] -= 1; // last boundary no longer maxDoc
        bad.add(new Partitioned(m.segments, parts));
      }
      return bad;
    }
  }

  private class Partitioned extends MergePolicy.OneMerge {
    final int[][] parts;

    Partitioned(List<SegmentCommitInfo> segments, int[][] parts) {
      super(segments);
      this.parts = parts;
    }

    @Override
    public boolean isPartitioned() {
      return true;
    }

    @Override
    public int[][] getDocRangePartitions(List<CodecReader> readers) {
      return parts;
    }

    @Override
    public CodecReader wrapForMerge(CodecReader reader) throws IOException {
      // Runs after initMergeReaders has snapshotted liveDocs, so anything the
      // test deletes from here on is genuinely concurrent.
      mergeStarted.countDown();
      try {
        proceed.await(30, TimeUnit.SECONDS);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException(e);
      }
      return reader;
    }
  }
}
