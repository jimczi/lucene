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
import org.apache.lucene.codecs.PointsFormat;
import org.apache.lucene.codecs.PointsReader;
import org.apache.lucene.codecs.PointsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;

/**
 * Stores a segment's points as one sub-index per contiguous range of documents, rather than as one
 * tree over all of them.
 *
 * <p>Points are the mirror image of a terms dictionary here. A dictionary <i>shares keys</i>: one
 * entry for a term serves every range, so splitting it writes the term bytes and the block-tree
 * index once per range, which is pure duplication. A block k-d tree shares nothing -- every {@code
 * (value, doc)} pair belongs to exactly one document -- so splitting it only redistributes pairs.
 * The extra cost is the per-tree structure, and with {@link
 * org.apache.lucene.util.bkd.BKDConfig#DEFAULT_MAX_POINTS_IN_LEAF_NODE} points to a leaf the split
 * values are a small fraction of the data, plus one partly filled leaf per range.
 *
 * <p>Sharing also <i>hurts</i> what this is for. One tree is ordered by value, so the documents of
 * every range are interleaved through every leaf: answering a range query for one small range
 * traverses leaves full of other ranges' documents and discards almost all of them, costing what
 * the whole segment costs rather than what that range costs.
 *
 * <p>And it is what makes a partitioned merge affordable. An output of such a merge keeps only the
 * documents of its own range and maps the rest to {@code -1}; with one tree per segment it must
 * still read all of the points to find that out, so a k-output merge reads them k times. Stored per
 * range, an output skips the sub-indexes that hold it nothing -- see {@link
 * PerRangePointsWriter#merge}.
 *
 * @lucene.experimental
 */
public final class PerRangePointsFormat extends PointsFormat {

  /** Name of this format, as written into the segment metadata. */
  public static final String NAME = "PerRangePoints";

  static final String META_CODEC = "PerRangePointsMeta";
  static final String META_EXTENSION = "prpm";
  static final int VERSION_START = 0;
  static final int VERSION_CURRENT = VERSION_START;

  private final PointsFormat delegate;
  private final DocRanges ranges;

  /**
   * @param delegate format each range's own points are written with
   * @param ranges how a segment's documents are cut into ranges
   */
  public PerRangePointsFormat(PointsFormat delegate, DocRanges ranges) {
    this.delegate = delegate;
    this.ranges = ranges;
  }

  /**
   * The document boundaries of one segment's ranges.
   *
   * <p>Ranges are intervals of document id, which is what makes them free to skip: under an index
   * sort on the routing key, a key range <i>is</i> a document range. Implementations must return
   * {@code numRanges + 1} non-decreasing boundaries starting at 0 and ending at {@code maxDoc}.
   */
  @FunctionalInterface
  public interface DocRanges {
    /** Boundaries for a segment with {@code maxDoc} documents. */
    int[] boundaries(int maxDoc);
  }

  /** Cuts a segment into {@code numRanges} equal spans of document id. */
  public static DocRanges equalSpans(int numRanges) {
    return maxDoc -> {
      int[] bounds = new int[numRanges + 1];
      for (int r = 1; r <= numRanges; r++) {
        bounds[r] = (int) ((long) maxDoc * r / numRanges);
      }
      return bounds;
    };
  }

  /** The segment suffix range {@code r}'s own points are written under. */
  static String rangeSuffix(String outerSuffix, int range) {
    // SegmentWriteState allows at most one underscore in a suffix, so the marker is appended.
    return outerSuffix + "R" + range;
  }

  @Override
  public PointsWriter fieldsWriter(SegmentWriteState state) throws IOException {
    return new PerRangePointsWriter(delegate, ranges, state);
  }

  @Override
  public PointsReader fieldsReader(SegmentReadState state) throws IOException {
    return new PerRangePointsReader(delegate, state);
  }
}
