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

package org.apache.lucene.codecs.hnsw;

import java.io.IOException;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.apache.lucene.util.quantization.RandomAccessQuantizedByteVectorValues;
import org.apache.lucene.util.quantization.ScalarQuantizedRandomVectorScorer;

/** Scalar quantized implementation of {@link RandomVectorScorer}. */
public class ScalarQuantizedVectorScorerSupplier implements RandomVectorScorerSupplier {
  private final RandomVectorScorerSupplier nonQuantizedDelegate;

  public ScalarQuantizedVectorScorerSupplier(RandomVectorScorerSupplier nonQuantizedDelegate) {
    this.nonQuantizedDelegate = nonQuantizedDelegate;
  }

  @Override
  public RandomVectorScorer create(
      RandomAccessVectorValues values, VectorSimilarityFunction similarityFunction)
      throws IOException {
    if (values instanceof RandomAccessQuantizedByteVectorValues quantizedByteVectorValues) {
      return new ScalarQuantizedRandomVectorScorer(
          similarityFunction,
          quantizedByteVectorValues.getScalarQuantizer(),
          quantizedByteVectorValues,
          null);
    }
    return nonQuantizedDelegate.create(values, similarityFunction);
  }

  @Override
  public RandomVectorScorer create(
      RandomAccessVectorValues values, VectorSimilarityFunction similarityFunction, float[] query)
      throws IOException {
    if (values instanceof RandomAccessQuantizedByteVectorValues quantizedByteVectorValues) {
      return new ScalarQuantizedRandomVectorScorer(
          similarityFunction,
          quantizedByteVectorValues.getScalarQuantizer(),
          quantizedByteVectorValues,
          query);
    }
    return nonQuantizedDelegate.create(values, similarityFunction, query);
  }

  @Override
  public RandomVectorScorer create(
      RandomAccessVectorValues values, VectorSimilarityFunction similarityFunction, byte[] query)
      throws IOException {
    throw new IllegalArgumentException("scalar quantization does not support byte[] targets");
  }

  @Override
  public String toString() {
    return "ScalarQuantizedVectorScorerSupplier("
        + "nonQuantizedDelegate="
        + nonQuantizedDelegate
        + ')';
  }
}
