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

package org.apache.lucene.util.hnsw;

import java.io.IOException;
import org.apache.lucene.util.Bits;

/** A {@link RandomVectorScorer} for scoring random nodes in batches against an abstract query. */
public abstract class RandomVectorScorer {
  private final RandomAccessVectorValues values;

  public RandomVectorScorer(RandomAccessVectorValues values) {
    this.values = values;
  }

  /**
   * Make a copy of the supplier, which will copy the underlying vectorValues so the copy is safe to
   * be used in other threads.
   */
  public abstract RandomVectorScorer copy() throws IOException;

  /**
   * @return the maximum possible ordinal for this scorer
   */
  public int maxOrd() {
    return values.size();
  }

  /**
   * Translates vector ordinal to the correct document ID. By default, this is an identity function.
   *
   * @param ord the vector ordinal
   * @return the document Id for that vector ordinal
   */
  public int ordToDoc(int ord) {
    return values.ordToDoc(ord);
  }

  /**
   * Returns the {@link Bits} representing live documents. By default, this is an identity function.
   *
   * @param acceptDocs the accept docs
   * @return the accept docs
   */
  public Bits getAcceptOrds(Bits acceptDocs) {
    return values.getAcceptOrds(acceptDocs);
  }

  /**
   * Switch the query to the provided {@code ord}. All scores will be computed against this ordinal
   * after this call.
   *
   * @param ord the ordinal of the query vector
   * @return the modified random vector scorer
   */
  public abstract RandomVectorScorer setQueryOrd(int ord) throws IOException;

  /**
   * Returns the score between the query and the provided node.
   *
   * @param node a random node in the graph
   * @return the computed score
   */
  public abstract float score(int node) throws IOException;
}
