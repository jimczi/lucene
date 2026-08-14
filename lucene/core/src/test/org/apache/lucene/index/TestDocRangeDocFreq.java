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
import java.util.List;
import java.util.Locale;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.BytesRef;

/**
 * {@link TermsEnum#docFreq(int,int)} must agree with counting the postings by hand, and must
 * actually separate one group of documents from another -- which is the reason it exists.
 */
public class TestDocRangeDocFreq extends LuceneTestCase {

  private static final String FIELD = "body";
  private static final String TENANT = "tenant";

  /** Counts by walking every posting, which is the answer the method has to reproduce. */
  private static int bruteForce(TermsEnum termsEnum, int minDoc, int maxDoc) throws IOException {
    final PostingsEnum postings = termsEnum.postings(null, PostingsEnum.NONE);
    int count = 0;
    for (int doc = postings.nextDoc();
        doc != DocIdSetIterator.NO_MORE_DOCS;
        doc = postings.nextDoc()) {
      if (doc >= minDoc && doc < maxDoc) {
        count++;
      }
    }
    return count;
  }

  public void testMatchesBruteForceCount() throws IOException {
    try (Directory dir = newDirectory()) {
      final int numDocs = atLeast(600);
      try (IndexWriter writer =
          new IndexWriter(dir, newIndexWriterConfig(new MockAnalyzer(random())))) {
        for (int i = 0; i < numDocs; i++) {
          final Document doc = new Document();
          final StringBuilder body = new StringBuilder();
          // A handful of terms, so some are dense enough to span blocks and some are sparse.
          for (int t = 0; t < 6; t++) {
            body.append('t').append(random().nextInt(40)).append(' ');
          }
          doc.add(new TextField(FIELD, body.toString(), Field.Store.NO));
          writer.addDocument(doc);
        }
        writer.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        final LeafReader leaf = getOnlyLeafReader(reader);
        final Terms terms = leaf.terms(FIELD);
        assertNotNull(terms);
        for (int iter = 0; iter < 200; iter++) {
          final int minDoc = random().nextInt(leaf.maxDoc());
          final int maxDoc = minDoc + random().nextInt(leaf.maxDoc() - minDoc + 1);
          final BytesRef term = new BytesRef("t" + random().nextInt(40));

          TermsEnum termsEnum = terms.iterator();
          if (termsEnum.seekExact(term) == false) {
            continue;
          }
          final int expected = bruteForce(termsEnum, minDoc, maxDoc);
          // A fresh enum, so the count cannot depend on the brute-force walk having run.
          termsEnum = terms.iterator();
          assertTrue(termsEnum.seekExact(term));
          assertEquals(
              "term " + term.utf8ToString() + " over [" + minDoc + "," + maxDoc + ")",
              expected,
              termsEnum.docFreq(minDoc, maxDoc));
        }
      }
    }
  }

  public void testWholeRangeMatchesDocFreq() throws IOException {
    try (Directory dir = newDirectory()) {
      try (IndexWriter writer =
          new IndexWriter(dir, newIndexWriterConfig(new MockAnalyzer(random())))) {
        for (int i = 0; i < 300; i++) {
          final Document doc = new Document();
          doc.add(new TextField(FIELD, "common t" + (i % 7), Field.Store.NO));
          writer.addDocument(doc);
        }
        writer.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        final LeafReader leaf = getOnlyLeafReader(reader);
        final TermsEnum termsEnum = leaf.terms(FIELD).iterator();
        while (termsEnum.next() != null) {
          assertEquals(
              "the whole document space must agree with the plain statistic",
              termsEnum.docFreq(),
              termsEnum.docFreq(0, leaf.maxDoc()));
        }
      }
    }
  }

  /**
   * The point of the method: a term that is rare in the segment but common in one tenant reports
   * the tenant's own frequency, not the segment's.
   */
  public void testSeparatesTenants() throws IOException {
    try (Directory dir = newDirectory()) {
      final IndexWriterConfig config = newIndexWriterConfig(new MockAnalyzer(random()));
      // Groups only become document ranges once the sort has put each tenant's documents together.
      config.setIndexSort(new Sort(new SortField(TENANT, SortField.Type.STRING)));
      final List<String> tenants = List.of("a", "b", "c");
      try (IndexWriter writer = new IndexWriter(dir, config)) {
        for (String tenant : tenants) {
          for (int i = 0; i < 100; i++) {
            final Document doc = new Document();
            doc.add(new StringField(TENANT, tenant, Field.Store.NO));
            doc.add(new SortedDocValuesField(TENANT, new BytesRef(tenant)));
            // "jargon" is in every document of tenant b, and in none of the others.
            final String body = tenant.equals("b") ? "shared jargon" : "shared";
            doc.add(new TextField(FIELD, body, Field.Store.NO));
            writer.addDocument(doc);
          }
        }
        writer.forceMerge(1);
      }
      try (DirectoryReader reader = DirectoryReader.open(dir)) {
        final LeafReader leaf = getOnlyLeafReader(reader);
        final List<int[]> ranges = tenantRanges(leaf, tenants);

        final TermsEnum termsEnum = leaf.terms(FIELD).iterator();
        assertTrue(termsEnum.seekExact(new BytesRef("jargon")));
        assertEquals(100, termsEnum.docFreq());

        final int[] rangeA = ranges.get(0);
        final int[] rangeB = ranges.get(1);
        final int[] rangeC = ranges.get(2);
        assertEquals(0, termsEnum.docFreq(rangeA[0], rangeA[1]));
        // Every one of tenant b's documents, so within b the term carries no discriminating power
        // at all -- the opposite of what the segment-wide statistic says about it.
        assertEquals(100, termsEnum.docFreq(rangeB[0], rangeB[1]));
        assertEquals(0, termsEnum.docFreq(rangeC[0], rangeC[1]));
      }
    }
  }

  /** Where each tenant's documents start and end, read off the sorted index. */
  private static List<int[]> tenantRanges(LeafReader leaf, List<String> tenants)
      throws IOException {
    final List<int[]> ranges = new ArrayList<>();
    for (String tenant : tenants) {
      final TermsEnum termsEnum = leaf.terms(TENANT).iterator();
      assertTrue(termsEnum.seekExact(new BytesRef(tenant)));
      final PostingsEnum postings = termsEnum.postings(null, PostingsEnum.NONE);
      int first = -1;
      int last = -1;
      for (int doc = postings.nextDoc();
          doc != DocIdSetIterator.NO_MORE_DOCS;
          doc = postings.nextDoc()) {
        if (first == -1) {
          first = doc;
        }
        last = doc;
      }
      final int[] range = new int[] {first, last + 1};
      assertEquals(
          String.format(
              Locale.ROOT, "tenant %s should occupy one contiguous run of documents", tenant),
          100,
          range[1] - range[0]);
      ranges.add(range);
    }
    return ranges;
  }
}
