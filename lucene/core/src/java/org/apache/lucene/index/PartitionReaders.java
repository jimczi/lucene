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

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.Version;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

/**
 * Opens a read-only view over a <em>subset</em> of a commit's segments — the segments accepted by a
 * predicate — <b>without opening any other segment</b>. Combined with one-segment-per-partition writing
 * (see {@link DocumentPartitioner}), this yields a reader scoped to a single partition (tenant/slice):
 * <ul>
 *   <li>only that partition's segments are opened, so an inactive partition's data is never loaded;</li>
 *   <li>the reader has its own doc-id space {@code [0, sum-of-accepted-maxDoc)}, independent of the rest of
 *       the index, so a partition is not constrained by other partitions' sizes.</li>
 * </ul>
 * This is the read-side counterpart to Lucene's usual whole-commit {@link DirectoryReader#open}, which
 * eagerly opens every segment.
 *
 * @lucene.experimental
 */
public final class PartitionReaders {

    private PartitionReaders() {}

    /**
     * Returns a {@link DirectoryReader} over exactly the segments of {@code commit} for which
     * {@code segmentFilter} returns true, opening no others. The returned reader owns and will close the
     * opened segment readers. If no segment matches, an empty reader is returned. It is a real
     * {@code DirectoryReader} (see {@link PartitionDirectoryReader}), so it can be used wherever the
     * {@code DirectoryReader} contract is required.
     */
    public static DirectoryReader open(Directory directory, IndexCommit commit, Predicate<SegmentCommitInfo> segmentFilter)
        throws IOException {
        final SegmentInfos infos = SegmentInfos.readCommit(directory, commit.getSegmentsFileName());
        final List<LeafReader> readers = new ArrayList<>();
        boolean success = false;
        try {
            for (SegmentCommitInfo sci : infos) {
                if (segmentFilter.test(sci)) {
                    readers.add(new SegmentReader(sci, infos.getIndexCreatedVersionMajor(), IOContext.DEFAULT));
                }
            }
            final DirectoryReader reader = new PartitionDirectoryReader(directory, readers.toArray(new LeafReader[0]), infos, commit);
            success = true;
            return reader;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(readers);
            }
        }
    }

    /**
     * Returns a {@link DirectoryReader} over an <em>explicit</em> list of segments, opening no others and
     * requiring no enclosing commit. This is for reading a partition whose segments have been {@link
     * IndexWriter#detachPartition detached} from the live commit (e.g. an evicted/idle tenant): the caller
     * holds the segments' {@link SegmentCommitInfo}s (as returned by {@code detachPartition}) and their files
     * are still on disk, so the partition stays readable without re-attaching it into the writer. The
     * returned reader has no {@link DirectoryReader#getIndexCommit() index commit} (it is not a point in the
     * directory's commit history). The caller owns and must close it.
     */
    public static DirectoryReader openSegments(Directory directory, List<SegmentCommitInfo> segments) throws IOException {
        final SegmentInfos infos = new SegmentInfos(Version.LATEST.major);
        for (SegmentCommitInfo sci : segments) {
            infos.add(sci);
        }
        final List<LeafReader> readers = new ArrayList<>();
        boolean success = false;
        try {
            for (SegmentCommitInfo sci : segments) {
                readers.add(new SegmentReader(sci, infos.getIndexCreatedVersionMajor(), IOContext.DEFAULT));
            }
            final DirectoryReader reader =
                new PartitionDirectoryReader(directory, readers.toArray(new LeafReader[0]), infos, null);
            success = true;
            return reader;
        } finally {
            if (success == false) {
                IOUtils.closeWhileHandlingException(readers);
            }
        }
    }
}
