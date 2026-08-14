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
package org.apache.lucene.codecs;

import java.io.Closeable;
import java.io.IOException;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergePolicy;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.PointValues;

/**
 * Abstract API to visit point values.
 *
 * @lucene.experimental
 */
public abstract class PointsReader implements Closeable {

  /** Sole constructor. (For invocation by subclass constructors, typically implicit.) */
  protected PointsReader() {}

  /**
   * Checks consistency of this reader.
   *
   * <p>Note that this may be costly in terms of I/O, e.g. may involve computing a checksum value
   * against large data files. A {@code OneMerge} can be provided so that expensive checksum
   * computations can be periodically interrupted when the merge is aborted.
   *
   * @param merge the merge to check for abort, or {@code null} for non-interruptible behavior
   */
  public abstract void checkIntegrity(MergePolicy.OneMerge merge) throws IOException;

  /**
   * Return {@link PointValues} for the given {@code field}. The behavior is undefined if the given
   * field doesn't have points enabled on its {@link FieldInfo}.
   */
  public abstract PointValues getValues(String field);

  /**
   * Returns an instance optimized for merging. This instance may only be used in the thread that
   * acquires it.
   *
   * <p>The default implementation returns {@code this}
   */
  public PointsReader getMergeInstance() {
    return this;
  }

  /**
   * Returns an instance optimized for merging into an output that keeps only the documents {@code
   * docMap} maps to a document, discarding the rest. This instance may only be used in the thread
   * that acquires it, and is not closed separately from the reader it came from.
   *
   * <p>A merge that splits its input into several outputs calls this once per output, so a reader
   * holding its points in per-document-range pieces can hand back only the pieces an output still
   * wants and leave the others unread. Without it such a merge reads all of the points once per
   * output to discover that most of them map nowhere, because points are ordered by value rather
   * than by document and an output's documents are therefore spread through every leaf.
   *
   * <p>The default implementation cannot know how its points are laid out, so it ignores the map
   * and returns {@link #getMergeInstance()}.
   */
  public PointsReader getMergeInstance(MergeState.DocMap docMap) {
    return getMergeInstance();
  }
}
