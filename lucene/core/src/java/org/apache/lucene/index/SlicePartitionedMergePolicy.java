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
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A {@link MergePolicy} that keeps merges <b>within a single partition</b> (tenant/slice), preserving
 * the one-segment-per-partition invariant produced by a partition-sticky indexing buffer (see {@link
 * IndexWriterConfig#setDocumentPartitioner}). It groups the segments by their {@link
 * DocumentPartitioner#PARTITION_ATTRIBUTE} and asks the wrapped delegate to select merges independently
 * per group, so no proposed merge ever mixes two partitions. Each resulting merged segment is stamped
 * with its partition so it stays grouped for subsequent merges. Segments with no partition attribute
 * (e.g. written without a partitioner) form a single default group and merge exactly as the delegate
 * would.
 *
 * <p>This is the companion piece that makes partitioned buffering usable: without it a normal policy
 * would merge across partitions and re-mix tenants. It uses only public {@link MergePolicy} API plus the
 * public {@link DocumentPartitioner#PARTITION_ATTRIBUTE} constant, so it can equally live outside Lucene
 * (e.g. supplied by the host via {@link IndexWriterConfig#setMergePolicy}).
 *
 * @lucene.experimental
 */
public final class SlicePartitionedMergePolicy extends FilterMergePolicy {

  /** Wraps {@code in} (e.g. a {@link TieredMergePolicy}), constraining every merge to one partition. */
  public SlicePartitionedMergePolicy(MergePolicy in) {
    super(in);
  }

  @Override
  public MergeSpecification findMerges(
      MergeTrigger mergeTrigger, SegmentInfos segmentInfos, MergeContext mergeContext)
      throws IOException {
    MergeSpecification spec = null;
    for (Map.Entry<String, List<SegmentCommitInfo>> group : groupByPartition(segmentInfos).entrySet()) {
      if (group.getValue().size() > 1) {
        final MergeSpecification sub =
            in.findMerges(mergeTrigger, subInfos(segmentInfos, group.getValue()), mergeContext);
        spec = appendTagged(spec, sub, group.getKey());
      }
    }
    return spec;
  }

  @Override
  public MergeSpecification findForcedMerges(
      SegmentInfos segmentInfos,
      int maxSegmentCount,
      Map<SegmentCommitInfo, Boolean> segmentsToMerge,
      MergeContext mergeContext)
      throws IOException {
    final Map<String, List<SegmentCommitInfo>> byPartition = groupByPartition(segmentInfos);
    // Distribute the requested segment budget across partitions; a partition cannot go below 1 segment.
    final int perPartition = Math.max(1, maxSegmentCount / Math.max(1, byPartition.size()));
    MergeSpecification spec = null;
    for (Map.Entry<String, List<SegmentCommitInfo>> group : byPartition.entrySet()) {
      final Map<SegmentCommitInfo, Boolean> groupToMerge = new HashMap<>();
      for (SegmentCommitInfo sci : group.getValue()) {
        final Boolean requested = segmentsToMerge.get(sci);
        if (requested != null) {
          groupToMerge.put(sci, requested);
        }
      }
      if (groupToMerge.isEmpty() == false) {
        final MergeSpecification sub =
            in.findForcedMerges(
                subInfos(segmentInfos, group.getValue()), perPartition, groupToMerge, mergeContext);
        spec = appendTagged(spec, sub, group.getKey());
      }
    }
    return spec;
  }

  @Override
  public MergeSpecification findFullFlushMerges(
      MergeTrigger mergeTrigger, SegmentInfos segmentInfos, MergeContext mergeContext)
      throws IOException {
    MergeSpecification spec = null;
    for (Map.Entry<String, List<SegmentCommitInfo>> group : groupByPartition(segmentInfos).entrySet()) {
      if (group.getValue().size() > 1) {
        final MergeSpecification sub =
            in.findFullFlushMerges(mergeTrigger, subInfos(segmentInfos, group.getValue()), mergeContext);
        spec = appendTagged(spec, sub, group.getKey());
      }
    }
    return spec;
  }

  @Override
  public MergeSpecification findForcedDeletesMerges(
      SegmentInfos segmentInfos, MergeContext mergeContext) throws IOException {
    MergeSpecification spec = null;
    for (Map.Entry<String, List<SegmentCommitInfo>> group : groupByPartition(segmentInfos).entrySet()) {
      final MergeSpecification sub =
          in.findForcedDeletesMerges(subInfos(segmentInfos, group.getValue()), mergeContext);
      spec = appendTagged(spec, sub, group.getKey());
    }
    return spec;
  }

  private static Map<String, List<SegmentCommitInfo>> groupByPartition(SegmentInfos segmentInfos) {
    final Map<String, List<SegmentCommitInfo>> byPartition = new LinkedHashMap<>();
    for (SegmentCommitInfo sci : segmentInfos.asList()) {
      final String key = sci.info.getAttribute(DocumentPartitioner.PARTITION_ATTRIBUTE);
      byPartition.computeIfAbsent(key == null ? "" : key, k -> new ArrayList<>()).add(sci);
    }
    return byPartition;
  }

  /** A {@link SegmentInfos} view holding only {@code group}'s (original) commit infos. */
  private static SegmentInfos subInfos(SegmentInfos all, List<SegmentCommitInfo> group) {
    final SegmentInfos sub = all.clone();
    sub.clear();
    for (SegmentCommitInfo sci : group) {
      sub.add(sci);
    }
    return sub;
  }

  /** Adds the delegate's merges to {@code target}, tagging each merged segment with {@code partitionKey}. */
  private static MergeSpecification appendTagged(
      MergeSpecification target, MergeSpecification toAdd, String partitionKey) {
    if (toAdd == null || toAdd.merges.isEmpty()) {
      return target;
    }
    if (target == null) {
      target = new MergeSpecification();
    }
    for (OneMerge merge : toAdd.merges) {
      target.add(partitionKey.isEmpty() ? merge : partitionTagged(merge, partitionKey));
    }
    return target;
  }

  /**
   * Wraps a delegate merge so the resulting merged segment records its partition (via {@link
   * SegmentInfo#putAttribute}), keeping it in the right group for future merges.
   */
  private static OneMerge partitionTagged(OneMerge delegate, String partitionKey) {
    return new OneMerge(delegate.segments) {
      @Override
      public void setMergeInfo(SegmentCommitInfo info) {
        info.info.putAttribute(DocumentPartitioner.PARTITION_ATTRIBUTE, partitionKey);
        super.setMergeInfo(info);
      }
    };
  }
}
