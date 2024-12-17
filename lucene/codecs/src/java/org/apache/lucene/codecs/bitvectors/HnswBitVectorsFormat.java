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

package org.apache.lucene.codecs.bitvectors;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_NUM_MERGE_WORKER;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.MAXIMUM_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.MAXIMUM_MAX_CONN;

import java.io.IOException;
import java.util.concurrent.ExecutorService;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.TaskExecutor;
import org.apache.lucene.store.ReadAdvice;
import org.apache.lucene.util.hnsw.HnswGraph;

/**
 * Encodes bit vector values into an associated graph connecting the documents having values. The
 * graph is used to power HNSW search. The format consists of two files, and uses {@link
 * Lucene99FlatVectorsFormat} to store the actual vectors, but with a custom scorer implementation:
 * For details on graph storage and file extensions, see {@link Lucene99HnswVectorsFormat}.
 *
 * @lucene.experimental
 */
public final class HnswBitVectorsFormat extends KnnVectorsFormat {

  public static final String NAME = "HnswBitVectorsFormat";

  /**
   * Controls how many of the nearest neighbor candidates are connected to the new node. Defaults to
   * {@link Lucene99HnswVectorsFormat#DEFAULT_MAX_CONN}. See {@link HnswGraph} for more details.
   */
  private final int maxConn;

  /**
   * The number of candidate neighbors to track while searching the graph for each newly inserted
   * node. Defaults to {@link Lucene99HnswVectorsFormat#DEFAULT_BEAM_WIDTH}. See {@link HnswGraph}
   * for details.
   */
  private final int beamWidth;

  /** The format for storing, reading, merging vectors on disk */
  private final FlatVectorsFormat flatVectorsFormat;

  private final int numMergeWorkers;
  private final TaskExecutor mergeExec;

  /** Constructs a format using default graph construction parameters */
  public HnswBitVectorsFormat() {
    this(DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH, DEFAULT_NUM_MERGE_WORKER, null);
  }

  /**
   * Constructs a format using the given graph construction parameters.
   *
   * @param maxConn the maximum number of connections to a node in the HNSW graph
   * @param beamWidth the size of the queue maintained during graph construction.
   */
  public HnswBitVectorsFormat(int maxConn, int beamWidth) {
    this(maxConn, beamWidth, DEFAULT_NUM_MERGE_WORKER, null);
  }

  /**
   * Constructs a format using the given graph construction parameters and scalar quantization.
   *
   * @param maxConn the maximum number of connections to a node in the HNSW graph
   * @param beamWidth the size of the queue maintained during graph construction.
   * @param numMergeWorkers number of workers (threads) that will be used when doing merge. If
   *     larger than 1, a non-null {@link ExecutorService} must be passed as mergeExec
   * @param mergeExec the {@link ExecutorService} that will be used by ALL vector writers that are
   *     generated by this format to do the merge
   */
  public HnswBitVectorsFormat(
      int maxConn, int beamWidth, int numMergeWorkers, ExecutorService mergeExec) {
    super(NAME);
    if (maxConn <= 0 || maxConn > MAXIMUM_MAX_CONN) {
      throw new IllegalArgumentException(
          "maxConn must be positive and less than or equal to "
              + MAXIMUM_MAX_CONN
              + "; maxConn="
              + maxConn);
    }
    if (beamWidth <= 0 || beamWidth > MAXIMUM_BEAM_WIDTH) {
      throw new IllegalArgumentException(
          "beamWidth must be positive and less than or equal to "
              + MAXIMUM_BEAM_WIDTH
              + "; beamWidth="
              + beamWidth);
    }
    this.maxConn = maxConn;
    this.beamWidth = beamWidth;
    if (numMergeWorkers == 1 && mergeExec != null) {
      throw new IllegalArgumentException(
          "No executor service is needed as we'll use single thread to merge");
    }
    this.numMergeWorkers = numMergeWorkers;
    if (mergeExec != null) {
      this.mergeExec = new TaskExecutor(mergeExec);
    } else {
      this.mergeExec = null;
    }
    this.flatVectorsFormat =
        new Lucene99FlatVectorsFormat(new FlatBitVectorsScorer(), ReadAdvice.RANDOM);
  }

  @Override
  public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new FlatBitVectorsWriter(
        new Lucene99HnswVectorsWriter(
            state,
            maxConn,
            beamWidth,
            flatVectorsFormat.fieldsWriter(state),
            numMergeWorkers,
            mergeExec));
  }

  @Override
  public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
    return new Lucene99HnswVectorsReader(state, flatVectorsFormat.fieldsReader(state));
  }

  @Override
  public int getMaxDimensions(String fieldName) {
    return 1024;
  }

  @Override
  public String toString() {
    return "HnswBitVectorsFormat(name=HnswBitVectorsFormat, maxConn="
        + maxConn
        + ", beamWidth="
        + beamWidth
        + ", flatVectorFormat="
        + flatVectorsFormat
        + ")";
  }

  private static class FlatBitVectorsWriter extends KnnVectorsWriter {
    private final KnnVectorsWriter delegate;

    public FlatBitVectorsWriter(KnnVectorsWriter delegate) {
      this.delegate = delegate;
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
      delegate.mergeOneField(fieldInfo, mergeState);
    }

    @Override
    public void finish() throws IOException {
      delegate.finish();
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
      if (fieldInfo.getVectorEncoding() != VectorEncoding.BYTE) {
        throw new IllegalArgumentException("HnswBitVectorsFormat only supports BYTE encoding");
      }
      return delegate.addField(fieldInfo);
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
      delegate.flush(maxDoc, sortMap);
    }

    @Override
    public void close() throws IOException {
      delegate.close();
    }

    @Override
    public long ramBytesUsed() {
      return delegate.ramBytesUsed();
    }
  }
}
