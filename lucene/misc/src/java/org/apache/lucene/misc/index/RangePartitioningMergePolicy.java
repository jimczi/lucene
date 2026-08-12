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
package org.apache.lucene.misc.index;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.FilterMergePolicy;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeTrigger;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.BytesRefBuilder;

/**
 * Rewrites segments so that each output owns a contiguous range of the index sort key, using {@link
 * MergePolicy.OneMerge#getDocRangePartitions()}.
 *
 * <p>Without this, every segment holds an arbitrary subset of keys, so a query restricted to one
 * key must open nearly every segment. Merging cannot fix that by <em>selection</em> -- merging any
 * k segments reduces the spread by the same amount, because the key sets are unstructured subsets.
 * It has to be fixed by <em>splitting</em>: rewriting the inputs so each output owns one key range.
 * Afterwards a key lives in one segment per level, and ordinary merges of range-adjacent segments
 * preserve that.
 *
 * <p>Boundaries are placed on real key values rather than on document counts: the policy walks the
 * union of the inputs' dictionaries for {@code field} in order, accumulating document counts, and
 * cuts every {@code totalDocs/outputs} documents. Cutting per-segment document counts instead would
 * give each segment slightly different boundaries and leave keys straddling two outputs.
 *
 * <p>The walk is a streaming k-way merge holding one term per input, so its memory is proportional
 * to the number of input segments rather than to the number of distinct keys.
 *
 * <p>{@code field} must be the primary index sort field and indexed as {@link SortedDocValues}.
 *
 * @lucene.experimental
 */
public class RangePartitioningMergePolicy extends FilterMergePolicy {

  private final String field;
  private final int outputs;
  private final int minSegmentsToPartition;

  /** Segments already partitioned, so the work is not repeated. */
  private final Set<String> partitioned = new HashSet<>();

  /**
   * @param in the policy to delegate to when no partitioning is due
   * @param field primary index sort field, indexed as SortedDocValues
   * @param outputs number of key ranges to produce
   * @param minSegmentsToPartition only partition once this many unpartitioned segments exist
   */
  public RangePartitioningMergePolicy(
      MergePolicy in, String field, int outputs, int minSegmentsToPartition) {
    super(in);
    if (outputs < 2) {
      throw new IllegalArgumentException("outputs must be >= 2, got " + outputs);
    }
    this.field = field;
    this.outputs = outputs;
    this.minSegmentsToPartition = minSegmentsToPartition;
  }

  @Override
  public MergeSpecification findMerges(MergeTrigger trigger, SegmentInfos infos, MergeContext ctx)
      throws IOException {
    List<SegmentCommitInfo> candidates = new ArrayList<>();
    for (SegmentCommitInfo si : infos) {
      if (ctx.getMergingSegments().contains(si)) {
        // Stay out of the way while anything is already merging.
        return super.findMerges(trigger, infos, ctx);
      }
      if (partitioned.contains(si.info.name) == false) {
        candidates.add(si);
      }
    }
    if (candidates.size() < minSegmentsToPartition) {
      return super.findMerges(trigger, infos, ctx);
    }
    MergeSpecification spec = new MergeSpecification();
    spec.add(new RangeMerge(candidates, field, outputs, partitioned));
    return spec;
  }

  private static class RangeMerge extends MergePolicy.OneMerge {
    private final String field;
    private final int outputs;
    private final Set<String> partitioned;

    RangeMerge(
        List<SegmentCommitInfo> segments, String field, int outputs, Set<String> partitioned) {
      super(segments);
      this.field = field;
      this.outputs = outputs;
      this.partitioned = partitioned;
    }

    @Override
    public boolean isPartitioned() {
      return true;
    }

    @Override
    public int[][] getDocRangePartitions() throws IOException {
      final List<CodecReader> readers = getMergeReaders();

      // ord -> first docID, one linear pass per reader. A key occupies one
      // contiguous docID interval because the index is sorted by `field`.
      final int[][] starts = new int[readers.size()][];
      final SortedDocValues[] dvs = new SortedDocValues[readers.size()];
      long totalDocs = 0;
      for (int i = 0; i < readers.size(); i++) {
        starts[i] = firstDocPerOrd(readers.get(i), field);
        dvs[i] = readers.get(i).getSortedDocValues(field);
        totalDocs += readers.get(i).numDocs();
      }

      // Streaming k-way merge over the dictionaries, cutting every
      // totalDocs/outputs documents on a key boundary.
      final long per = Math.max(1, totalDocs / outputs);
      final int[] cursor = new int[readers.size()];
      final List<BytesRef> cuts = new ArrayList<>();
      long acc = 0;
      while (cuts.size() < outputs - 1) {
        BytesRef min = null;
        for (int i = 0; i < readers.size(); i++) {
          if (dvs[i] == null || cursor[i] >= dvs[i].getValueCount()) {
            continue;
          }
          BytesRef cand = dvs[i].lookupOrd(cursor[i]);
          if (min == null || cand.compareTo(min) < 0) {
            BytesRefBuilder copy = new BytesRefBuilder();
            copy.copyBytes(cand);
            min = copy.toBytesRef();
          }
        }
        if (min == null) {
          break; // dictionaries exhausted
        }
        if (acc >= per * (cuts.size() + 1)) {
          cuts.add(min);
        }
        for (int i = 0; i < readers.size(); i++) {
          if (dvs[i] == null || cursor[i] >= dvs[i].getValueCount()) {
            continue;
          }
          if (dvs[i].lookupOrd(cursor[i]).compareTo(min) == 0) {
            acc += starts[i][cursor[i] + 1] - starts[i][cursor[i]];
            cursor[i]++;
          }
        }
      }

      final int actual = cuts.size() + 1;
      final int[][] partitions = new int[readers.size()][actual + 1];
      for (int i = 0; i < readers.size(); i++) {
        partitions[i][0] = 0;
        for (int c = 0; c < cuts.size(); c++) {
          partitions[i][c + 1] = dvs[i] == null ? 0 : docOffsetOf(dvs[i], starts[i], cuts.get(c));
        }
        partitions[i][actual] = readers.get(i).maxDoc();
        // A key absent from this reader can make two cuts land on the same
        // offset; an empty output range is legal, a decreasing one is not.
        for (int o = 1; o <= actual; o++) {
          if (partitions[i][o] < partitions[i][o - 1]) {
            partitions[i][o] = partitions[i][o - 1];
          }
        }
      }

      for (SegmentCommitInfo si : segments) {
        partitioned.add(si.info.name);
      }
      return partitions;
    }

    private static int[] firstDocPerOrd(CodecReader r, String field) throws IOException {
      SortedDocValues dv = r.getSortedDocValues(field);
      int k = dv == null ? 0 : dv.getValueCount();
      int[] starts = new int[k + 1];
      java.util.Arrays.fill(starts, -1);
      if (dv != null) {
        for (int doc = dv.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = dv.nextDoc()) {
          int ord = dv.ordValue();
          if (starts[ord] == -1) {
            starts[ord] = doc;
          }
        }
      }
      starts[k] = r.maxDoc();
      for (int i = k - 1; i >= 0; i--) {
        if (starts[i] == -1) {
          starts[i] = starts[i + 1];
        }
      }
      return starts;
    }

    private static int docOffsetOf(SortedDocValues dv, int[] starts, BytesRef key)
        throws IOException {
      int ord = dv.lookupTerm(key);
      if (ord < 0) {
        ord = -ord - 1; // insertion point: first key >= the cut
      }
      if (ord >= starts.length - 1) {
        return starts[starts.length - 1];
      }
      return starts[ord];
    }
  }
}
