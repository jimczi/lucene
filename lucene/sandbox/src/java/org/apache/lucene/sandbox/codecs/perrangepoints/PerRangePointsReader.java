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
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.PointsFormat;
import org.apache.lucene.codecs.PointsReader;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.ChecksumIndexInput;
import org.apache.lucene.util.IOUtils;

/** Reads what {@link PerRangePointsWriter} wrote: one sub-index per range of documents. */
final class PerRangePointsReader extends PointsReader {

  private final PointsReader[] subs;
  private final int[] bounds;

  /** False for the view {@link #survivingOnly} hands out, which borrows another reader's files. */
  private final boolean ownsSubs;

  PerRangePointsReader(PointsFormat delegate, SegmentReadState state) throws IOException {
    final String metaName =
        IndexFileNames.segmentFileName(
            state.segmentInfo.name, state.segmentSuffix, PerRangePointsFormat.META_EXTENSION);
    int[] readBounds;
    boolean[] present;
    try (ChecksumIndexInput meta = state.directory.openChecksumInput(metaName)) {
      Throwable priorE = null;
      int[] b = null;
      boolean[] p = null;
      try {
        CodecUtil.checkIndexHeader(
            meta,
            PerRangePointsFormat.META_CODEC,
            PerRangePointsFormat.VERSION_START,
            PerRangePointsFormat.VERSION_CURRENT,
            state.segmentInfo.getId(),
            state.segmentSuffix);
        final int numRanges = meta.readVInt();
        if (numRanges <= 0) {
          throw new CorruptIndexException("invalid range count " + numRanges, meta);
        }
        b = new int[numRanges + 1];
        for (int r = 0; r <= numRanges; r++) {
          b[r] = meta.readVInt();
        }
        p = new boolean[numRanges];
        for (int r = 0; r < numRanges; r++) {
          p[r] = meta.readByte() != 0;
        }
      } catch (Throwable t) {
        priorE = t;
      } finally {
        CodecUtil.checkFooter(meta, priorE);
      }
      readBounds = b;
      present = p;
    }
    this.bounds = readBounds;

    final PointsReader[] opened = new PointsReader[present.length];
    boolean success = false;
    try {
      for (int r = 0; r < present.length; r++) {
        if (present[r]) {
          opened[r] =
              delegate.fieldsReader(
                  new SegmentReadState(
                      state, PerRangePointsFormat.rangeSuffix(state.segmentSuffix, r)));
        }
      }
      success = true;
    } finally {
      if (success == false) {
        IOUtils.closeWhileHandlingException(opened);
      }
    }
    this.subs = opened;
    this.ownsSubs = true;
  }

  private PerRangePointsReader(PointsReader[] subs, int[] bounds, boolean ownsSubs) {
    this.subs = subs;
    this.bounds = bounds;
    this.ownsSubs = ownsSubs;
  }

  /**
   * The same segment with only the ranges that still hold a document after {@code docMap}, for a
   * merge output that keeps some ranges and discards the rest.
   *
   * <p>This is the whole reason points are stored per range: the discarded ranges are never read.
   * With one tree over the segment an output would have to read all of the points before finding
   * out that it wanted none of them, so a merge into k outputs read them k times.
   *
   * <p>The returned reader shares this one's sub-readers and must not be closed.
   */
  PointsReader survivingOnly(MergeState.DocMap docMap) {
    final PointsReader[] kept = new PointsReader[subs.length];
    for (int r = 0; r < subs.length; r++) {
      if (subs[r] == null) {
        continue;
      }
      for (int doc = bounds[r]; doc < bounds[r + 1]; doc++) {
        if (docMap.get(doc) >= 0) {
          kept[r] = subs[r];
          break;
        }
      }
    }
    return new PerRangePointsReader(kept, bounds, false);
  }

  @Override
  public PointValues getValues(String field) {
    final List<PointValues> present = new ArrayList<>(subs.length);
    for (PointsReader sub : subs) {
      if (sub == null) {
        continue;
      }
      final PointValues values = sub.getValues(field);
      if (values != null) {
        present.add(values);
      }
    }
    if (present.isEmpty()) {
      return null;
    }
    if (present.size() == 1) {
      return present.get(0);
    }
    return new UnionPointValues(present.toArray(PointValues[]::new));
  }

  @Override
  public void checkIntegrity(MergePolicy.OneMerge merge) throws IOException {
    for (PointsReader sub : subs) {
      if (sub != null) {
        sub.checkIntegrity(merge);
      }
    }
  }

  @Override
  public void close() throws IOException {
    if (ownsSubs) {
      IOUtils.close(subs);
    }
  }
}
