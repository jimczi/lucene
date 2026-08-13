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
package org.apache.lucene.sandbox.codecs.perrangepoints;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.IntPoint;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.BytesRef;

/**
 * The per-range points format must answer exactly what one tree over the same documents answers.
 * Splitting the tree changes where points live, never which documents match.
 */
public class TestPerRangePointsFormat extends LuceneTestCase {

  private static final String POINT_FIELD = "value";
  private static final String ROUTING_FIELD = "routing";

  private record Doc(String routing, int value) {}

  private static List<Doc> randomDocs(int count) {
    final List<Doc> docs = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      // Few tenants relative to documents, so ranges hold many documents each.
      docs.add(
          new Doc(
              String.format(Locale.ROOT, "t%03d", random().nextInt(64)), random().nextInt(10_000)));
    }
    return docs;
  }

  private static Document toDocument(Doc doc) {
    final Document d = new Document();
    d.add(new StringField(ROUTING_FIELD, doc.routing(), Field.Store.NO));
    d.add(new SortedDocValuesField(ROUTING_FIELD, new BytesRef(doc.routing())));
    d.add(new IntPoint(POINT_FIELD, doc.value()));
    return d;
  }

  private static Codec perRangeCodec() {
    return new PerRangePointsTestCodec();
  }

  private static IndexWriterConfig config(Codec codec) {
    final IndexWriterConfig c = new IndexWriterConfig(null);
    c.setCodec(codec);
    // Ranges are intervals of document id, so they only mean anything under the routing sort.
    c.setIndexSort(new Sort(new SortField(ROUTING_FIELD, SortField.Type.STRING)));
    return c;
  }

  private static void index(Directory dir, Codec codec, List<Doc> docs, boolean forceMerge)
      throws IOException {
    try (IndexWriter writer = new IndexWriter(dir, config(codec))) {
      for (int i = 0; i < docs.size(); i++) {
        writer.addDocument(toDocument(docs.get(i)));
        if ((i + 1) % 40 == 0) {
          writer.flush();
        }
      }
      writer.commit();
      if (forceMerge) {
        writer.forceMerge(1);
      }
    }
  }

  private void assertSameAsStock(boolean forceMerge) throws IOException {
    final List<Doc> docs = randomDocs(atLeast(500));
    try (Directory stockDir = newDirectory();
        Directory rangedDir = newDirectory()) {
      index(stockDir, new Lucene104Codec(), docs, forceMerge);
      index(rangedDir, perRangeCodec(), docs, forceMerge);

      try (DirectoryReader stock = DirectoryReader.open(stockDir);
          DirectoryReader ranged = DirectoryReader.open(rangedDir)) {
        assertEquals(stock.numDocs(), ranged.numDocs());
        final IndexSearcher stockSearcher = new IndexSearcher(stock);
        final IndexSearcher rangedSearcher = new IndexSearcher(ranged);
        for (int iter = 0; iter < 40; iter++) {
          final int lo = random().nextInt(10_000);
          final int hi = lo + random().nextInt(2_000);
          final Query query = IntPoint.newRangeQuery(POINT_FIELD, lo, hi);
          assertEquals(
              "range [" + lo + "," + hi + "] disagreed",
              stockSearcher.count(query),
              rangedSearcher.count(query));
        }
        // An exact-value query exercises the single-leaf path rather than a wide traversal.
        for (int iter = 0; iter < 20; iter++) {
          final int value = random().nextInt(10_000);
          final Query query = IntPoint.newExactQuery(POINT_FIELD, value);
          assertEquals(stockSearcher.count(query), rangedSearcher.count(query));
        }
      }
    }
  }

  public void testMatchesStockAcrossSegments() throws IOException {
    assertSameAsStock(false);
  }

  public void testMatchesStockAfterMerge() throws IOException {
    assertSameAsStock(true);
  }
}
