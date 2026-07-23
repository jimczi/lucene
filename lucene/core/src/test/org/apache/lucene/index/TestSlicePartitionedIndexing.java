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

import java.util.HashSet;
import java.util.Set;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.util.LuceneTestCase;

/**
 * Proves that a {@link DocumentPartitioner} makes {@link IndexWriter} buffer documents per partition
 * and flush <b>one segment per partition</b> — the write-path foundation for physically isolating a
 * tenant/slice (so it can be loaded, evicted, or encrypted independently).
 */
public class TestSlicePartitionedIndexing extends LuceneTestCase {

  private static final String[] SLICES = {"tenantA", "tenantB", "tenantC"};

  /** Routes each document to a buffer keyed by its {@code slice} field value. */
  private static IndexWriterConfig partitionedBySlice() {
    IndexWriterConfig iwc = new IndexWriterConfig(new MockAnalyzer(random()));
    // Disable merging so we observe exactly what flush produced.
    iwc.setMergePolicy(NoMergePolicy.INSTANCE);
    iwc.setDocumentPartitioner(
        doc -> {
          for (IndexableField f : doc) {
            if (f.name().equals("slice")) {
              return f.stringValue();
            }
          }
          return null;
        });
    return iwc;
  }

  private static Document doc(String slice, int i) {
    Document d = new Document();
    d.add(new StringField("slice", slice, Field.Store.YES));
    d.add(new StringField("id", slice + "-" + i, Field.Store.NO));
    return d;
  }

  public void testOneSlicePerSegmentOnFlush() throws Exception {
    try (Directory dir = newDirectory()) {
      try (IndexWriter w = new IndexWriter(dir, partitionedBySlice())) {
        // Interleave slices so an unpartitioned writer would mix them into shared segments.
        final int perSlice = 50;
        for (int i = 0; i < perSlice; i++) {
          for (String slice : SLICES) {
            w.addDocument(doc(slice, i));
          }
        }
        w.commit();
      }

      try (DirectoryReader r = DirectoryReader.open(dir)) {
        assertTrue("expected multiple segments, got " + r.leaves().size(), r.leaves().size() >= SLICES.length);
        final Set<String> slicesSeen = new HashSet<>();
        for (LeafReaderContext ctx : r.leaves()) {
          final LeafReader lr = ctx.reader();
          final StoredFields stored = lr.storedFields();
          final Set<String> inSegment = new HashSet<>();
          for (int d = 0; d < lr.maxDoc(); d++) {
            inSegment.add(stored.document(d).get("slice"));
          }
          assertEquals("each segment must hold exactly one slice, found " + inSegment, 1, inSegment.size());
          slicesSeen.addAll(inSegment);
        }
        assertEquals("every slice must be represented", Set.of(SLICES), slicesSeen);
      }
    }
  }

  /** Set of partition keys that currently have a buffered (in-memory) DWPT. */
  private static Set<Object> bufferedPartitions(IndexWriter w) {
    Set<Object> keys = new HashSet<>();
    for (DocumentsWriterPerThread dwpt : w.docWriter.perThreadPool) {
      if (dwpt.getNumDocsInRAM() > 0) {
        keys.add(dwpt.partitionKey);
      }
    }
    return keys;
  }

  public void testFlushSliceFlushesOnlyThatSlice() throws Exception {
    try (Directory dir = newDirectory()) {
      try (IndexWriter w = new IndexWriter(dir, partitionedBySlice())) {
        for (String slice : SLICES) {
          w.addDocument(doc(slice, 0));
        }
        assertEquals("all three slices buffered", Set.of((Object) "tenantA", "tenantB", "tenantC"), bufferedPartitions(w));

        assertTrue("tenantA had a buffer to flush", w.flushSlice("tenantA"));
        assertEquals("only tenantA was flushed", Set.of((Object) "tenantB", "tenantC"), bufferedPartitions(w));

        assertFalse("nothing left to flush for tenantA", w.flushSlice("tenantA"));
      }
    }
  }

  public void testMaxActivePartitionsEvictsLruSlice() throws Exception {
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = partitionedBySlice().setMaxActivePartitions(2);
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        w.addDocument(doc("tenantA", 0));
        w.addDocument(doc("tenantB", 0));
        assertEquals("within the cap, both buffered", Set.of((Object) "tenantA", "tenantB"), bufferedPartitions(w));

        // Third active partition exceeds the cap -> the LRU partition (tenantA) is flushed out.
        w.addDocument(doc("tenantC", 0));
        assertEquals("LRU slice evicted, working set stays bounded", Set.of((Object) "tenantB", "tenantC"), bufferedPartitions(w));
      }
    }
  }

  private static Set<String> slicesInSegment(LeafReader lr) throws Exception {
    final StoredFields stored = lr.storedFields();
    final Set<String> inSegment = new HashSet<>();
    for (int d = 0; d < lr.maxDoc(); d++) {
      inSegment.add(stored.document(d).get("slice"));
    }
    return inSegment;
  }

  public void testSliceAwareMergeNeverMixesSlicesAndForceMergeIsPerSlice() throws Exception {
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = new IndexWriterConfig(new MockAnalyzer(random()));
      iwc.setDocumentPartitioner(
          doc -> {
            for (IndexableField f : doc) {
              if (f.name().equals("slice")) {
                return f.stringValue();
              }
            }
            return null;
          });
      // Slice-aware policy wrapping the standard tiered policy: merges stay within a slice.
      iwc.setMergePolicy(new SlicePartitionedMergePolicy(new TieredMergePolicy()));

      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        // Many flushes → many small single-slice segments that the policy may merge (within a slice).
        for (int round = 0; round < 8; round++) {
          for (String slice : SLICES) {
            for (int i = 0; i < 4; i++) {
              w.addDocument(doc(slice, round * 4 + i));
            }
          }
          w.flush();
        }

        // Even mid-flight, no natural merge may have mixed two slices.
        try (DirectoryReader r = DirectoryReader.open(w)) {
          for (LeafReaderContext ctx : r.leaves()) {
            assertEquals("a merged segment mixed slices", 1, slicesInSegment(ctx.reader()).size());
          }
        }

        // forceMerge(1) cannot go below one segment per slice (cross-slice merges are forbidden).
        w.forceMerge(1);
      }

      try (DirectoryReader r = DirectoryReader.open(dir)) {
        assertEquals("forceMerge(1) collapses to exactly one segment per slice", SLICES.length, r.leaves().size());
        final Set<String> slicesSeen = new HashSet<>();
        for (LeafReaderContext ctx : r.leaves()) {
          final Set<String> inSegment = slicesInSegment(ctx.reader());
          assertEquals("each final segment holds exactly one slice", 1, inSegment.size());
          slicesSeen.addAll(inSegment);
        }
        assertEquals(Set.of(SLICES), slicesSeen);
      }
    }
  }

  /** Control: with no partitioner, the same interleaved docs land in shared (mixed) segments. */
  public void testUnpartitionedMixesSlices() throws Exception {
    try (Directory dir = newDirectory()) {
      IndexWriterConfig iwc = new IndexWriterConfig(new MockAnalyzer(random()));
      iwc.setMergePolicy(NoMergePolicy.INSTANCE);
      try (IndexWriter w = new IndexWriter(dir, iwc)) {
        for (int i = 0; i < 50; i++) {
          for (String slice : SLICES) {
            w.addDocument(doc(slice, i));
          }
        }
        w.commit();
      }
      try (DirectoryReader r = DirectoryReader.open(dir)) {
        boolean sawMixedSegment = false;
        for (LeafReaderContext ctx : r.leaves()) {
          final LeafReader lr = ctx.reader();
          final StoredFields stored = lr.storedFields();
          final Set<String> inSegment = new HashSet<>();
          for (int d = 0; d < lr.maxDoc(); d++) {
            inSegment.add(stored.document(d).get("slice"));
          }
          sawMixedSegment |= inSegment.size() > 1;
        }
        assertTrue("without a partitioner at least one segment should mix slices", sawMixedSegment);
      }
    }
  }
}
