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
import org.apache.lucene.util.IOUtils;

import java.io.IOException;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArraySet;
import java.util.concurrent.ExecutorService;

/**
 * A {@link DirectoryReader} over a <em>subset</em> of a commit's segments (see {@link PartitionReaders}). Unlike
 * {@link StandardDirectoryReader} it opens only the accepted segments, giving a real {@code DirectoryReader} — with
 * its own {@link CacheHelper core cache key}, {@link #getIndexCommit() commit}, and {@link #getVersion() version} —
 * scoped to a single partition (tenant/slice). Being a {@code DirectoryReader} (not a bare {@link MultiReader}) is
 * what lets it flow through consumers that require the {@code DirectoryReader} contract (e.g. Elasticsearch's
 * per-shard reader wrapping and cache-key checks).
 *
 * <p>Reopen is not supported directly ({@link #doOpenIfChanged} report "no change"); the owner reopens by opening a
 * fresh partition reader against a newer commit.
 *
 * @lucene.experimental
 */
public final class PartitionDirectoryReader extends DirectoryReader {

    private final SegmentInfos segmentInfos;
    private final IndexCommit commit;

    private final Set<ClosedListener> readerClosedListeners = new CopyOnWriteArraySet<>();
    private final CacheHelper cacheHelper = new CacheHelper() {
        private final CacheKey cacheKey = new CacheKey();

        @Override
        public CacheKey getKey() {
            return cacheKey;
        }

        @Override
        public void addClosedListener(ClosedListener listener) {
            ensureOpen();
            readerClosedListeners.add(listener);
        }
    };

    PartitionDirectoryReader(Directory directory, LeafReader[] readers, SegmentInfos segmentInfos, IndexCommit commit)
        throws IOException {
        super(directory, readers, null);
        this.segmentInfos = segmentInfos;
        this.commit = commit;
    }

    @Override
    public long getVersion() {
        ensureOpen();
        return segmentInfos.getVersion();
    }

    @Override
    public boolean isCurrent() throws IOException {
        ensureOpen();
        return SegmentInfos.readLatestCommit(directory()).getVersion() == segmentInfos.getVersion();
    }

    @Override
    public IndexCommit getIndexCommit() throws IOException {
        ensureOpen();
        return commit;
    }

    // Reopen is managed by the owner (it opens a fresh partition reader on a newer commit); report "no change".
    @Override
    protected DirectoryReader doOpenIfChanged() {
        return null;
    }

    @Override
    protected DirectoryReader doOpenIfChanged(ExecutorService executorService) {
        return null;
    }

    @Override
    protected DirectoryReader doOpenIfChanged(IndexCommit commit) {
        return null;
    }

    @Override
    protected DirectoryReader doOpenIfChanged(IndexCommit commit, ExecutorService executorService) {
        return null;
    }

    @Override
    protected DirectoryReader doOpenIfChanged(IndexWriter writer, boolean applyAllDeletes) {
        return null;
    }

    @Override
    protected DirectoryReader doOpenIfChanged(IndexWriter writer, boolean applyAllDeletes, ExecutorService executorService) {
        return null;
    }

    @Override
    protected void doClose() throws IOException {
        IOUtils.applyToAll(getSequentialSubReaders(), LeafReader::decRef);
    }

    @Override
    protected void notifyReaderClosedListeners() throws IOException {
        synchronized (readerClosedListeners) {
            IOUtils.applyToAll(readerClosedListeners, l -> l.onClose(cacheHelper.getKey()));
        }
    }

    @Override
    public CacheHelper getReaderCacheHelper() {
        return cacheHelper;
    }
}
