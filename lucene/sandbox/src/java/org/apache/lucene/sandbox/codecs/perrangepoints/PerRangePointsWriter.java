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
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.PointsFormat;
import org.apache.lucene.codecs.PointsReader;
import org.apache.lucene.codecs.PointsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.IOUtils;

/** Writes one sub-index per range of documents, plus the metadata a reader needs to find them. */
final class PerRangePointsWriter extends PointsWriter {

  private final PointsFormat delegate;
  private final SegmentWriteState state;
  private final int[] bounds;
  private final PointsWriter[] rangeWriters;

  PerRangePointsWriter(
      PointsFormat delegate, PerRangePointsFormat.DocRanges ranges, SegmentWriteState state) {
    this.delegate = delegate;
    this.state = state;
    final int maxDoc = state.segmentInfo.maxDoc();
    this.bounds = ranges.boundaries(maxDoc);
    if (bounds.length < 2 || bounds[0] != 0 || bounds[bounds.length - 1] != maxDoc) {
      throw new IllegalArgumentException(
          "range boundaries must start at 0 and end at maxDoc=" + maxDoc);
    }
    for (int r = 1; r < bounds.length; r++) {
      if (bounds[r] < bounds[r - 1]) {
        throw new IllegalArgumentException("range boundaries must not decrease");
      }
    }
    this.rangeWriters = new PointsWriter[bounds.length - 1];
  }

  private PointsWriter writerFor(int range) throws IOException {
    if (rangeWriters[range] == null) {
      rangeWriters[range] =
          delegate.fieldsWriter(
              new SegmentWriteState(
                  state, PerRangePointsFormat.rangeSuffix(state.segmentSuffix, range)));
    }
    return rangeWriters[range];
  }

  @Override
  public void writeField(FieldInfo fieldInfo, PointsReader reader) throws IOException {
    for (int range = 0; range < rangeWriters.length; range++) {
      if (bounds[range] == bounds[range + 1]) {
        continue; // no documents, so nothing to write and no sub-index to open
      }
      writerFor(range)
          .writeField(fieldInfo, new DocRangeReader(reader, bounds[range], bounds[range + 1]));
    }
  }

  /**
   * Merges range by range, reading from each input only the ranges that still hold a document this
   * output keeps.
   *
   * <p>Everything else is the inherited merge: once the inputs are narrowed, Lucene's own per-field
   * merge writes the points, and {@link #writeField} splits the result across this output's ranges.
   */
  @Override
  public void merge(MergeState mergeState) throws IOException {
    // Each input is asked to narrow itself to what this output keeps. A reader that stores its
    // points per range hands back only the ranges still wanted; any other reader returns itself,
    // which is why this needs no test of what kind of reader it got.
    final PointsReader[] narrowed = new PointsReader[mergeState.pointsReaders.length];
    for (int i = 0; i < narrowed.length; i++) {
      final PointsReader in = mergeState.pointsReaders[i];
      narrowed[i] = in == null ? null : in.getMergeInstance(mergeState.docMaps[i]);
    }
    super.merge(
        new MergeState(
            mergeState.docMaps,
            mergeState.segmentInfo,
            mergeState.mergeFieldInfos,
            mergeState.storedFieldsReaders,
            mergeState.termVectorsReaders,
            mergeState.normsProducers,
            mergeState.docValuesProducers,
            mergeState.fieldInfos,
            mergeState.liveDocs,
            mergeState.fieldsProducers,
            narrowed,
            mergeState.knnVectorsReaders,
            mergeState.maxDocs,
            mergeState.infoStream,
            mergeState.intraMergeTaskExecutor,
            mergeState.needsIndexSort,
            mergeState.oneMerge));
  }

  @Override
  public void finish() throws IOException {
    for (PointsWriter writer : rangeWriters) {
      if (writer != null) {
        writer.finish();
      }
    }
    final String metaName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, PerRangePointsFormat.META_EXTENSION);
    try (IndexOutput meta = state.directory.createOutput(metaName, state.context)) {
      CodecUtil.writeIndexHeader(
          meta,
          PerRangePointsFormat.META_CODEC,
          PerRangePointsFormat.VERSION_CURRENT,
          state.segmentInfo.getId(),
          state.segmentSuffix);
      meta.writeVInt(rangeWriters.length);
      for (int bound : bounds) {
        meta.writeVInt(bound);
      }
      for (PointsWriter writer : rangeWriters) {
        meta.writeByte((byte) (writer == null ? 0 : 1));
      }
      CodecUtil.writeFooter(meta);
    }
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(rangeWriters);
  }

  /** One range's worth of a reader's points, as the delegate writer sees them. */
  private static final class DocRangeReader extends PointsReader {

    private final PointsReader in;
    private final int lo;
    private final int hi;

    DocRangeReader(PointsReader in, int lo, int hi) {
      this.in = in;
      this.lo = lo;
      this.hi = hi;
    }

    @Override
    public PointValues getValues(String field) {
      final PointValues values = in.getValues(field);
      return values == null ? null : new DocRangeValues(values, lo, hi);
    }

    @Override
    public void checkIntegrity(MergePolicy.OneMerge merge) throws IOException {
      in.checkIntegrity(merge);
    }

    @Override
    public void close() throws IOException {
      in.close();
    }
  }

  /**
   * A field's points restricted to a document range.
   *
   * <p>The tree is not rebuilt: it is walked as it is and the visits outside the range are dropped,
   * so {@link #size} and the bounding box are those of the whole field and describe more than the
   * caller will see. That is safe for a writer, which uses the count only to size its buffers.
   */
  private static final class DocRangeValues extends PointValues {

    private final PointValues in;
    private final int lo;
    private final int hi;

    DocRangeValues(PointValues in, int lo, int hi) {
      this.in = in;
      this.lo = lo;
      this.hi = hi;
    }

    @Override
    public PointTree getPointTree() throws IOException {
      return new DocRangeTree(in.getPointTree(), lo, hi);
    }

    @Override
    public byte[] getMinPackedValue() {
      return in.getMinPackedValue();
    }

    @Override
    public byte[] getMaxPackedValue() {
      return in.getMaxPackedValue();
    }

    @Override
    public int getNumDimensions() {
      return in.getNumDimensions();
    }

    @Override
    public int getNumIndexDimensions() {
      return in.getNumIndexDimensions();
    }

    @Override
    public int getBytesPerDimension() {
      return in.getBytesPerDimension();
    }

    @Override
    public long size() {
      return in.size();
    }

    @Override
    public int getDocCount() {
      return in.getDocCount();
    }
  }

  /** Delegates the walk, and drops the visits that fall outside the range. */
  private static final class DocRangeTree implements PointValues.PointTree {

    private final PointValues.PointTree in;
    private final int lo;
    private final int hi;

    DocRangeTree(PointValues.PointTree in, int lo, int hi) {
      this.in = in;
      this.lo = lo;
      this.hi = hi;
    }

    @Override
    public PointValues.PointTree clone() {
      return new DocRangeTree(in.clone(), lo, hi);
    }

    @Override
    public boolean moveToChild() throws IOException {
      return in.moveToChild();
    }

    @Override
    public boolean moveToSibling() throws IOException {
      return in.moveToSibling();
    }

    @Override
    public boolean moveToParent() throws IOException {
      return in.moveToParent();
    }

    @Override
    public byte[] getMinPackedValue() {
      return in.getMinPackedValue();
    }

    @Override
    public byte[] getMaxPackedValue() {
      return in.getMaxPackedValue();
    }

    @Override
    public long size() {
      return in.size();
    }

    @Override
    public void visitDocIDs(PointValues.IntersectVisitor visitor) throws IOException {
      in.visitDocIDs(filter(visitor));
    }

    @Override
    public void visitDocValues(PointValues.IntersectVisitor visitor) throws IOException {
      in.visitDocValues(filter(visitor));
    }

    private PointValues.IntersectVisitor filter(PointValues.IntersectVisitor visitor) {
      return new PointValues.IntersectVisitor() {
        @Override
        public void visit(int docID) throws IOException {
          if (docID >= lo && docID < hi) {
            visitor.visit(docID);
          }
        }

        @Override
        public void visit(int docID, byte[] packedValue) throws IOException {
          if (docID >= lo && docID < hi) {
            visitor.visit(docID, packedValue);
          }
        }

        @Override
        public PointValues.Relation compare(byte[] minPackedValue, byte[] maxPackedValue) {
          // A cell says nothing about which documents it holds, so it can never be skipped here.
          return visitor.compare(minPackedValue, maxPackedValue);
        }
      };
    }
  }
}
