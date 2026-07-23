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

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.Version;

/**
 * Proves the write-side of the "hold only the active partitions in memory" model: {@link
 * IndexWriter#detachPartition} removes a partition's segments from the live commit (so the writer no
 * longer carries them in its resident {@link SegmentInfos}) while retaining their files, and {@link
 * IndexWriter#attachPartition} restores them intact. In stateless this is what lets an idle tenant's
 * segments live only in the object store while the durable manifest keeps listing them.
 */
public class TestSlicePartitionDetach extends LuceneTestCase {

  private static IndexWriterConfig partitionedBySlice() {
    IndexWriterConfig iwc = new IndexWriterConfig(new MockAnalyzer(random()));
    iwc.setMergePolicy(NoMergePolicy.INSTANCE); // observe exactly what flush produced
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

  private static Set<String> slicesInSegment(LeafReader lr) throws Exception {
    final StoredFields stored = lr.storedFields();
    final Set<String> inSegment = new HashSet<>();
    for (int d = 0; d < lr.maxDoc(); d++) {
      inSegment.add(stored.document(d).get("slice"));
    }
    return inSegment;
  }

  private static Set<String> filesOf(IndexWriter w, String slice) throws Exception {
    final Set<String> files = new HashSet<>();
    try (DirectoryReader r = DirectoryReader.open(w)) {
      for (LeafReaderContext ctx : r.leaves()) {
        SegmentReader sr = (SegmentReader) ctx.reader();
        if (slice.equals(sr.getSegmentInfo().info.getAttribute(DocumentPartitioner.PARTITION_ATTRIBUTE))) {
          files.addAll(sr.getSegmentInfo().files());
        }
      }
    }
    return files;
  }

  public void testDetachRemovesPartitionFromCommitButRetainsFilesThenAttachRestores() throws Exception {
    try (Directory dir = newDirectory()) {
      try (IndexWriter w = new IndexWriter(dir, partitionedBySlice())) {
        final int perSlice = 20;
        for (int i = 0; i < perSlice; i++) {
          w.addDocument(doc("tenantA", i));
          w.addDocument(doc("tenantB", i));
        }
        w.flush(); // one segment per slice

        final Set<String> aFiles = filesOf(w, "tenantA");
        assertFalse("tenantA must have its own segment files", aFiles.isEmpty());

        // Detaching an unknown partition is a no-op.
        assertTrue(w.detachPartition("nope").isEmpty());
        // Attaching one that was never detached is a no-op.
        assertEquals(0, w.attachPartition("tenantA"));

        // Detach tenantA: it leaves the live commit; a commit afterwards must NOT delete its files.
        assertFalse("tenantA had segments to detach", w.detachPartition("tenantA").isEmpty());
        w.commit();

        try (DirectoryReader r = DirectoryReader.open(dir)) {
          for (LeafReaderContext ctx : r.leaves()) {
            assertEquals("tenantA is gone from the commit", Set.of("tenantB"), slicesInSegment(ctx.reader()));
          }
        }
        final Set<String> onDisk = new HashSet<>(Arrays.asList(dir.listAll()));
        for (String f : aFiles) {
          assertTrue("detached tenantA file must be retained on disk: " + f, onDisk.contains(f));
        }

        // Re-attach: tenantA comes back with all its docs, still physically isolated in its own segment.
        assertTrue("tenantA was re-attached", w.attachPartition("tenantA") > 0);
        w.commit();

        final Set<String> slicesSeen = new HashSet<>();
        int aDocs = 0;
        try (DirectoryReader r = DirectoryReader.open(dir)) {
          for (LeafReaderContext ctx : r.leaves()) {
            final Set<String> inSegment = slicesInSegment(ctx.reader());
            assertEquals("each segment still holds exactly one slice", 1, inSegment.size());
            slicesSeen.addAll(inSegment);
            if (inSegment.equals(Set.of("tenantA"))) {
              aDocs += ctx.reader().maxDoc();
            }
          }
        }
        assertEquals(Set.of("tenantA", "tenantB"), slicesSeen);
        assertEquals("all of tenantA's docs restored", perSlice, aDocs);
      }
    }
  }

  /**
   * A detached partition can be persisted as a side commit ({@link SegmentInfos#writeToFile}) and later
   * reconstructed from that file alone ({@link SegmentInfos#readFromFile}) — the durable form needed to
   * reopen an evicted slice after the writing process is gone (segment names in a catalog are not enough:
   * Lucene needs the id/codec/delGen that only the serialized commit carries).
   */
  public void testDetachedSubsetPersistsAsSideCommitAndReopens() throws Exception {
    try (Directory dir = newDirectory()) {
      final long gen = 1L;
      final String sideCommit = "slice_tenantA.scommit";
      try (IndexWriter w = new IndexWriter(dir, partitionedBySlice())) {
        for (int i = 0; i < 12; i++) {
          w.addDocument(doc("tenantA", i));
          w.addDocument(doc("tenantB", i));
        }
        w.flush();

        final List<SegmentCommitInfo> detached = w.detachPartition("tenantA");
        assertFalse(detached.isEmpty());

        // Persist just tenantA's segments as a named side commit, then reconstruct from that file alone.
        final SegmentInfos subset = new SegmentInfos(Version.LATEST.major);
        for (SegmentCommitInfo sci : detached) {
          subset.add(sci);
        }
        subset.writeToFile(dir, sideCommit, gen);

        final SegmentInfos reloaded = SegmentInfos.readFromFile(dir, sideCommit, gen);
        try (DirectoryReader r = PartitionReaders.openSegments(dir, reloaded.asList())) {
          assertEquals("reconstructed reader sees tenantA's docs", 12, r.numDocs());
          final Set<String> seen = new HashSet<>();
          for (LeafReaderContext ctx : r.leaves()) {
            final StoredFields stored = ctx.reader().storedFields();
            for (int d = 0; d < ctx.reader().maxDoc(); d++) {
              seen.add(stored.document(d).get("slice"));
            }
          }
          assertEquals("reconstructed reader is isolated to tenantA", Set.of("tenantA"), seen);
        }
      }
    }
  }
}
