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

/**
 * Assigns a partition key to a document at index time so that {@link IndexWriter} routes it to a
 * dedicated in-memory indexing buffer ({@code DocumentsWriterPerThread}). All documents sharing a
 * partition key are buffered together and flushed into segments that contain <em>only</em> that
 * partition, so a segment maps one-to-one to a partition (e.g. a tenant/slice). This makes it
 * possible to load, evict, or encrypt a partition's segments independently.
 *
 * <p>When no partitioner is configured (the default), buffering is unpartitioned and behaves exactly
 * as before.
 *
 * @lucene.experimental
 */
@FunctionalInterface
public interface DocumentPartitioner {

  /**
   * {@link SegmentInfo} attribute under which a flushed segment records its partition key (the key's
   * {@link Object#toString()}). Because a partition-sticky buffer never mixes partitions, every segment
   * carries exactly one such value, which a slice-aware {@link MergePolicy} reads to keep merges within a
   * single partition.
   */
  String PARTITION_ATTRIBUTE = "lucene.partition.key";

  /**
   * Returns the partition key for {@code document}, or {@code null} for the default partition. Keys
   * are compared with {@link Object#equals}; prefer small immutable values (e.g. a {@link String} or
   * {@link org.apache.lucene.util.BytesRef}). For a document block (parent + children) this is
   * invoked on the first document only; the whole block shares its key.
   */
  Object partitionKey(Iterable<? extends IndexableField> document);
}
