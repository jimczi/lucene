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
package org.apache.lucene.util.quantization;

import java.io.IOException;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomVectorScorer;

/**
 * Quantized vector scorer
 *
 * @lucene.experimental
 */
public class ScalarQuantizedRandomVectorScorer extends RandomVectorScorer {
  private final VectorSimilarityFunction similarityFunction;
  private final ScalarQuantizedVectorSimilarity similarity;
  private final RandomAccessQuantizedByteVectorValues values;
  private final RandomAccessQuantizedByteVectorValues values1;
  private byte[] quantizedQuery;
  private float queryOffset;

  public ScalarQuantizedRandomVectorScorer(
      VectorSimilarityFunction similarityFunction,
      ScalarQuantizer scalarQuantizer,
      RandomAccessQuantizedByteVectorValues values,
      float[] query)
      throws IOException {
    super(values);
    this.similarityFunction = similarityFunction;
    this.similarity =
        ScalarQuantizedVectorSimilarity.fromVectorSimilarity(
            similarityFunction, scalarQuantizer.getConstantMultiplier(), scalarQuantizer.getBits());
    this.values = values;
    this.values1 = values.copy();
    if (query != null) {
      this.quantizedQuery = new byte[query.length];
      this.queryOffset = quantizeQuery(query, quantizedQuery, similarityFunction, scalarQuantizer);
    }
  }

  private ScalarQuantizedRandomVectorScorer(
      VectorSimilarityFunction similarityFunction,
      ScalarQuantizedVectorSimilarity similarity,
      RandomAccessQuantizedByteVectorValues values)
      throws IOException {
    super(values);
    this.similarityFunction = similarityFunction;
    this.similarity = similarity;
    this.values = values;
    this.values1 = values.copy();
  }

  @Override
  public RandomVectorScorer setQueryOrd(int ord) throws IOException {
    this.quantizedQuery = values.vectorValue(ord);
    this.queryOffset = values.getScoreCorrectionConstant();
    return this;
  }

  @Override
  public float score(int node) throws IOException {
    assert quantizedQuery != null;
    byte[] storedVectorValue = values1.vectorValue(node);
    float storedVectorCorrection = values1.getScoreCorrectionConstant();
    return similarity.score(
        quantizedQuery, this.queryOffset, storedVectorValue, storedVectorCorrection);
  }

  @Override
  public ScalarQuantizedRandomVectorScorer copy() throws IOException {
    return new ScalarQuantizedRandomVectorScorer(similarityFunction, similarity, values.copy());
  }

  public static float quantizeQuery(
      float[] query,
      byte[] quantizedQuery,
      VectorSimilarityFunction similarityFunction,
      ScalarQuantizer scalarQuantizer) {
    float[] processedQuery =
        switch (similarityFunction) {
          case EUCLIDEAN, DOT_PRODUCT, MAXIMUM_INNER_PRODUCT -> query;
          case COSINE -> {
            float[] queryCopy = ArrayUtil.copyOfSubArray(query, 0, query.length);
            VectorUtil.l2normalize(queryCopy);
            yield queryCopy;
          }
        };
    return scalarQuantizer.quantize(processedQuery, quantizedQuery, similarityFunction);
  }
}
