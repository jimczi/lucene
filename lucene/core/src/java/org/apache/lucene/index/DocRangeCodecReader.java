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
import java.util.Iterator;
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.codecs.PointsReader;
import org.apache.lucene.codecs.StoredFieldsReader;
import org.apache.lucene.codecs.TermVectorsReader;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;

/**
 * Exposes only documents in {@code [start, end)} of the wrapped reader, by treating everything
 * outside that range as deleted. Used by {@link IndexWriter} to build one output of a partitioned
 * merge (see {@link MergePolicy.OneMerge#getDocRangePartitions()}).
 *
 * <p>Documents outside the range map to {@code -1} in the resulting {@link MergeState.DocMap},
 * which is what lets the existing delete carry-over logic route each concurrently-arriving delete
 * to exactly the one output that owns the document, with no additional bookkeeping.
 *
 * <p>Integrity checks are suppressed here, because the merge has already run them once over these
 * same inputs. Each format begins its merge by checksumming every input file it is about to read,
 * which means reading the file in full; a partitioned merge runs those merges once per output, so
 * leaving the checks in place would checksum every input k times over. That is not a small share of
 * the merge -- verifying a file costs the same read the merge itself does -- and the k-1 repeats
 * cannot discover anything the first one did not.
 */
final class DocRangeCodecReader extends FilterCodecReader {

  private final Bits liveDocs;
  private final int numDocs;
  private final int start;
  private final int end;

  DocRangeCodecReader(CodecReader in, int start, int end) {
    super(in);
    this.start = start;
    this.end = end;
    assert start >= 0 && end <= in.maxDoc() && start <= end
        : "bad range [" + start + "," + end + ") maxDoc=" + in.maxDoc();
    FixedBitSet bits = new FixedBitSet(in.maxDoc());
    if (start < end) {
      // An output can legitimately own no document in this reader -- a key
      // missing here makes two cuts land on the same offset -- and
      // FixedBitSet#set rejects an empty range starting at maxDoc.
      bits.set(start, end);
    }
    Bits existing = in.getLiveDocs();
    if (existing != null) {
      existing.applyMask(bits, 0);
    }
    this.liveDocs = bits;
    this.numDocs = bits.cardinality();
  }

  @Override
  public Bits getLiveDocs() {
    return liveDocs;
  }

  @Override
  public int numDocs() {
    return numDocs;
  }

  /**
   * Doc values restricted to the range rather than merely masked.
   *
   * <p>Masking is enough for correctness but not for cost: a merge reads a field's values with
   * {@link DocValuesIterator#nextDoc()} and discards whatever the document map sends to {@code -1},
   * having already decoded it. Each output of a partitioned merge would therefore read every
   * document's values to keep its own share, and k outputs would read the segment k times. Seeking
   * to the range instead makes the outputs together read it once.
   */
  @Override
  public DocValuesProducer getDocValuesReader() {
    final DocValuesProducer values = in.getDocValuesReader();
    if (values == null) {
      return null;
    }
    return new DocRangeDocValuesProducer(values, start, end);
  }

  @Override
  public StoredFieldsReader getFieldsReader() {
    final StoredFieldsReader reader = in.getFieldsReader();
    return reader == null ? null : new PreverifiedStoredFieldsReader(reader);
  }

  @Override
  public TermVectorsReader getTermVectorsReader() {
    final TermVectorsReader reader = in.getTermVectorsReader();
    return reader == null ? null : new PreverifiedTermVectorsReader(reader);
  }

  @Override
  public NormsProducer getNormsReader() {
    final NormsProducer norms = in.getNormsReader();
    return norms == null ? null : new PreverifiedNormsProducer(norms);
  }

  @Override
  public PointsReader getPointsReader() {
    final PointsReader points = in.getPointsReader();
    return points == null ? null : new PreverifiedPointsReader(points);
  }

  @Override
  public FieldsProducer getPostingsReader() {
    final FieldsProducer postings = in.getPostingsReader();
    return postings == null ? null : new PreverifiedFieldsProducer(postings);
  }

  @Override
  public KnnVectorsReader getVectorReader() {
    final KnnVectorsReader vectors = in.getVectorReader();
    return vectors == null ? null : new PreverifiedKnnVectorsReader(vectors);
  }

  private static final class PreverifiedStoredFieldsReader extends StoredFieldsReader {
    private final StoredFieldsReader in;

    PreverifiedStoredFieldsReader(StoredFieldsReader in) {
      this.in = in;
    }

    @Override
    public void document(int docID, StoredFieldVisitor visitor) throws IOException {
      in.document(docID, visitor);
    }

    @Override
    public void prefetch(int docID) throws IOException {
      in.prefetch(docID);
    }

    @Override
    public StoredFieldsReader clone() {
      return new PreverifiedStoredFieldsReader(in.clone());
    }

    @Override
    public StoredFieldsReader getMergeInstance() {
      return new PreverifiedStoredFieldsReader(in.getMergeInstance());
    }

    @Override
    public void checkIntegrity(MergePolicy.OneMerge merge) {}

    @Override
    public void close() throws IOException {
      in.close();
    }
  }

  private static final class PreverifiedTermVectorsReader extends TermVectorsReader {
    private final TermVectorsReader in;

    PreverifiedTermVectorsReader(TermVectorsReader in) {
      this.in = in;
    }

    @Override
    public Fields get(int doc) throws IOException {
      return in.get(doc);
    }

    @Override
    public void prefetch(int docID) throws IOException {
      in.prefetch(docID);
    }

    @Override
    public TermVectorsReader clone() {
      return new PreverifiedTermVectorsReader(in.clone());
    }

    @Override
    public TermVectorsReader getMergeInstance() {
      return new PreverifiedTermVectorsReader(in.getMergeInstance());
    }

    @Override
    public void checkIntegrity(MergePolicy.OneMerge merge) {}

    @Override
    public void close() throws IOException {
      in.close();
    }
  }

  private static final class PreverifiedNormsProducer extends NormsProducer {
    private final NormsProducer in;

    PreverifiedNormsProducer(NormsProducer in) {
      this.in = in;
    }

    @Override
    public NumericDocValues getNorms(FieldInfo field) throws IOException {
      return in.getNorms(field);
    }

    @Override
    public NormsProducer getMergeInstance() {
      return new PreverifiedNormsProducer(in.getMergeInstance());
    }

    @Override
    public void checkIntegrity(MergePolicy.OneMerge merge) {}

    @Override
    public void close() throws IOException {
      in.close();
    }
  }

  private static final class PreverifiedPointsReader extends PointsReader {
    private final PointsReader in;

    PreverifiedPointsReader(PointsReader in) {
      this.in = in;
    }

    @Override
    public PointValues getValues(String field) {
      return in.getValues(field);
    }

    @Override
    public PointsReader getMergeInstance() {
      return new PreverifiedPointsReader(in.getMergeInstance());
    }

    @Override
    public PointsReader getMergeInstance(MergeState.DocMap docMap) {
      return new PreverifiedPointsReader(in.getMergeInstance(docMap));
    }

    @Override
    public void checkIntegrity(MergePolicy.OneMerge merge) {}

    @Override
    public void close() throws IOException {
      in.close();
    }
  }

  private static final class PreverifiedFieldsProducer extends FieldsProducer {
    private final FieldsProducer in;

    PreverifiedFieldsProducer(FieldsProducer in) {
      this.in = in;
    }

    @Override
    public Iterator<String> iterator() {
      return in.iterator();
    }

    @Override
    public Terms terms(String field) {
      return in.terms(field);
    }

    @Override
    public int size() {
      return in.size();
    }

    @Override
    public FieldsProducer getMergeInstance() {
      return new PreverifiedFieldsProducer(in.getMergeInstance());
    }

    @Override
    public void checkIntegrity(MergePolicy.OneMerge merge) {}

    @Override
    public void close() throws IOException {
      in.close();
    }
  }

  private static final class PreverifiedKnnVectorsReader extends KnnVectorsReader {
    private final KnnVectorsReader in;

    PreverifiedKnnVectorsReader(KnnVectorsReader in) {
      this.in = in;
    }

    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
      return in.getFloatVectorValues(field);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
      return in.getByteVectorValues(field);
    }

    @Override
    public Float16VectorValues getFloat16VectorValues(String field) throws IOException {
      return in.getFloat16VectorValues(field);
    }

    @Override
    public void search(String field, float[] target, KnnCollector collector, AcceptDocs acceptDocs)
        throws IOException {
      in.search(field, target, collector, acceptDocs);
    }

    @Override
    public void search(String field, byte[] target, KnnCollector collector, AcceptDocs acceptDocs)
        throws IOException {
      in.search(field, target, collector, acceptDocs);
    }

    @Override
    public void search(String field, short[] target, KnnCollector collector, AcceptDocs acceptDocs)
        throws IOException {
      in.search(field, target, collector, acceptDocs);
    }

    @Override
    public KnnVectorsReader getMergeInstance() throws IOException {
      return new PreverifiedKnnVectorsReader(in.getMergeInstance());
    }

    @Override
    public void finishMerge() throws IOException {
      in.finishMerge();
    }

    @Override
    public void checkIntegrity(MergePolicy.OneMerge merge) {}

    @Override
    public void close() throws IOException {
      in.close();
    }
  }

  @Override
  public CacheHelper getCoreCacheHelper() {
    return null;
  }

  @Override
  public CacheHelper getReaderCacheHelper() {
    return null;
  }
}
