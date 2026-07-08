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
package org.apache.lucene.search;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StoredField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FilterLeafReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.MultiReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.index.StoredFields;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.IOUtils;

/**
 * Exercises the search path when the composite reader's doc-id space extends beyond {@link
 * Integer#MAX_VALUE}, i.e. a hit's global doc id ({@link ScoreDoc#doc}) is a real {@code long}.
 *
 * <p>Indexing billions of documents is impractical, so this fakes the doc-id space: a few real
 * single-doc segments are wrapped in {@link FilterLeafReader}s that report an inflated {@code
 * maxDoc}. Placed before the "real" leaf in a {@link MultiReader}, they push that leaf's {@link
 * LeafReaderContext#docBase} — and therefore the doc ids of its hits — above {@link
 * Integer#MAX_VALUE}, without any large allocation (a {@link TermQuery} only ever touches the real
 * postings).
 */
public class TestSearchBeyondMaxInt extends LuceneTestCase {

  /** A leaf reader that claims to hold {@code maxDoc} (all-live) documents but delegates reads. */
  private static final class InflatedLeafReader extends FilterLeafReader {
    private final int inflatedMaxDoc;

    InflatedLeafReader(LeafReader in, int inflatedMaxDoc) {
      super(in);
      this.inflatedMaxDoc = inflatedMaxDoc;
    }

    @Override
    public int maxDoc() {
      return inflatedMaxDoc;
    }

    @Override
    public int numDocs() {
      // claim all inflated docs are live so numDeletedDocs()==0 stays consistent with a null liveDocs
      return inflatedMaxDoc;
    }

    @Override
    public Bits getLiveDocs() {
      return null;
    }

    @Override
    public CacheHelper getCoreCacheHelper() {
      return in.getCoreCacheHelper();
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
      return null; // maxDoc no longer matches the delegate: don't advertise as cacheable
    }
  }

  public void testHitsBeyondMaxInt() throws IOException {
    Directory dir = newDirectory();
    // Two padding segments (no "body" field) and one real segment with the matching docs.
    IndexWriterConfig iwc = new IndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE);
    final int numMatches = 3;
    try (IndexWriter w = new IndexWriter(dir, iwc)) {
      Document pad = new Document();
      pad.add(new StringField("pad", "x", Field.Store.NO));
      w.addDocument(pad);
      w.flush();
      w.addDocument(pad);
      w.flush();
      for (int i = 0; i < numMatches; i++) {
        Document doc = new Document();
        doc.add(new StringField("body", "match", Field.Store.NO));
        doc.add(new StoredField("id", i));
        w.addDocument(doc);
      }
      w.flush();
    }

    DirectoryReader dr = DirectoryReader.open(dir);
    assertEquals(3, dr.leaves().size());

    // Inflate the two padding leaves so the real leaf's docBase clears Integer.MAX_VALUE.
    final int inflated = IndexWriter.MAX_DOCS; // ~2.1B each; two of them => docBase > 2^31
    List<LeafReader> wrapped = new ArrayList<>();
    wrapped.add(new InflatedLeafReader(dr.leaves().get(0).reader(), inflated));
    wrapped.add(new InflatedLeafReader(dr.leaves().get(1).reader(), inflated));
    wrapped.add(dr.leaves().get(2).reader()); // real leaf, untouched

    MultiReader reader = new MultiReader(wrapped.toArray(new LeafReader[0]), false);
    try {
      // The composite doc-id space is now well beyond Integer.MAX_VALUE.
      long expectedDocBase = 2L * inflated;
      assertTrue(reader.totalMaxDoc() > Integer.MAX_VALUE);
      assertEquals(expectedDocBase + numMatches, reader.totalMaxDoc());
      List<LeafReaderContext> leaves = reader.leaves();
      assertEquals(expectedDocBase, leaves.get(2).docBase);
      // The legacy int accessor must refuse to represent a > 2^31 doc-id space.
      expectThrows(ArithmeticException.class, reader::maxDoc);

      IndexSearcher searcher = new IndexSearcher(reader);
      TopDocs topDocs = searcher.search(new TermQuery(new Term("body", "match")), numMatches);
      assertEquals(numMatches, topDocs.totalHits.value());
      assertEquals(numMatches, topDocs.scoreDocs.length);

      StoredFields storedFields = reader.storedFields();
      for (int i = 0; i < numMatches; i++) {
        ScoreDoc sd = topDocs.scoreDocs[i];
        // A genuine long doc id: docBase (> 2^31) + the leaf-local doc.
        assertTrue("doc id should exceed Integer.MAX_VALUE: " + sd.doc, sd.doc > Integer.MAX_VALUE);
        assertEquals(expectedDocBase + i, sd.doc);
        // Global-doc dispatch: resolve the leaf and read the right stored field back.
        assertEquals(2, ReaderUtil.subIndex(sd.doc, leaves));
        assertEquals(
            Integer.toString(i), storedFields.document(sd.doc).getField("id").numericValue() + "");
      }

      // Paging with a > 2^31 after.doc must work and continue past it.
      TopDocs page1 = searcher.search(new TermQuery(new Term("body", "match")), 1);
      assertEquals(expectedDocBase, page1.scoreDocs[0].doc);
      TopDocs page2 =
          searcher.searchAfter(page1.scoreDocs[0], new TermQuery(new Term("body", "match")), 1);
      assertEquals(1, page2.scoreDocs.length);
      assertEquals(expectedDocBase + 1, page2.scoreDocs[0].doc);

      // explain() must accept the long global doc id.
      Explanation exp =
          searcher.explain(new TermQuery(new Term("body", "match")), expectedDocBase + 1);
      assertTrue(exp.isMatch());
    } finally {
      IOUtils.close(reader, dr, dir);
    }
  }
}
