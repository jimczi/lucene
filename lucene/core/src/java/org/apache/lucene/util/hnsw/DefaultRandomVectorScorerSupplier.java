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
import org.apache.lucene.index.VectorSimilarityFunction;

/**
 * A {@link RandomVectorScorerSupplier} that computes scores using the default {@link
 * VectorSimilarityFunction}.
 */
public class DefaultRandomVectorScorerSupplier implements RandomVectorScorerSupplier {
  public static RandomVectorScorer createScorer(
      RandomAccessVectorValues values, VectorSimilarityFunction similarityFunction)
      throws IOException {
    return new DefaultRandomVectorScorerSupplier().create(values, similarityFunction);
  }

  public static RandomVectorScorer createScorer(
      RandomAccessVectorValues values, VectorSimilarityFunction similarityFunction, float[] query)
      throws IOException {
    return new DefaultRandomVectorScorerSupplier().create(values, similarityFunction, query);
  }

  public static RandomVectorScorer createScorer(
      RandomAccessVectorValues values, VectorSimilarityFunction similarityFunction, byte[] query)
      throws IOException {
    return new DefaultRandomVectorScorerSupplier().create(values, similarityFunction, query);
  }

  @Override
  public RandomVectorScorer create(
      RandomAccessVectorValues values, VectorSimilarityFunction similarityFunction)
      throws IOException {
    if (values instanceof RandomAccessVectorValues.Bytes bytes) {
      return new DefaultRandomVectorScorerSupplier.Bytes(bytes, similarityFunction);
    } else if (values instanceof RandomAccessVectorValues.Floats floats) {
      return new DefaultRandomVectorScorerSupplier.Floats(floats, similarityFunction);
    } else {
      throw new IllegalArgumentException("");
    }
  }

  @Override
  public RandomVectorScorer create(
      RandomAccessVectorValues values, VectorSimilarityFunction similarityFunction, float[] query)
      throws IOException {
    if (query.length != values.dimension()) {
      throw new IllegalArgumentException(
          "vector query dimension: "
              + query.length
              + " differs from field dimension: "
              + values.dimension());
    }
    if (values instanceof RandomAccessVectorValues.Floats floats) {
      var ret = new DefaultRandomVectorScorerSupplier.Floats(floats, similarityFunction);
      ret.setQuery(query);
      return ret;
    } else {
      throw new IllegalArgumentException("");
    }
  }

  @Override
  public RandomVectorScorer create(
      RandomAccessVectorValues values, VectorSimilarityFunction similarityFunction, byte[] query)
      throws IOException {
    if (query.length != values.dimension()) {
      throw new IllegalArgumentException(
          "vector query dimension: "
              + query.length
              + " differs from field dimension: "
              + values.dimension());
    }
    if (values instanceof RandomAccessVectorValues.Bytes bytes) {
      var ret = new DefaultRandomVectorScorerSupplier.Bytes(bytes, similarityFunction);
      ret.setQuery(query);
      return ret;
    } else {
      throw new IllegalArgumentException("");
    }
  }

  @Override
  public String toString() {
    return "DefaultRandomVectorScorerSupplier()";
  }

  /** FlatVectorScorer for Float vector */
  private final class Floats extends RandomVectorScorer {
    private final RandomAccessVectorValues.Floats vectors;
    private final RandomAccessVectorValues.Floats vectors1;
    private final RandomAccessVectorValues.Floats vectors2;
    private final VectorSimilarityFunction similarityFunction;
    private float[] query;

    private Floats(
        RandomAccessVectorValues.Floats vectors, VectorSimilarityFunction similarityFunction)
        throws IOException {
      super(vectors);
      this.vectors = vectors;
      vectors1 = vectors.copy();
      vectors2 = vectors.copy();
      this.similarityFunction = similarityFunction;
    }

    void setQuery(float[] query) {
      this.query = query;
    }

    @Override
    public RandomVectorScorer setQueryOrd(int ord) throws IOException {
      this.query = vectors1.vectorValue(ord);
      return this;
    }

    @Override
    public float score(int node) throws IOException {
      return similarityFunction.compare(query, vectors2.vectorValue(node));
    }

    @Override
    public RandomVectorScorer copy() throws IOException {
      return new DefaultRandomVectorScorerSupplier.Floats(vectors, similarityFunction);
    }
  }

  /** FlatVectorScorer for Float vector */
  private final class Bytes extends RandomVectorScorer {
    private final RandomAccessVectorValues.Bytes vectors;
    private final RandomAccessVectorValues.Bytes vectors1;
    private final RandomAccessVectorValues.Bytes vectors2;
    private final VectorSimilarityFunction similarityFunction;
    private byte[] query;

    private Bytes(
        RandomAccessVectorValues.Bytes vectors, VectorSimilarityFunction similarityFunction)
        throws IOException {
      super(vectors);
      this.vectors = vectors;
      vectors1 = vectors.copy();
      vectors2 = vectors.copy();
      this.similarityFunction = similarityFunction;
    }

    void setQuery(byte[] query) {
      this.query = query;
    }

    @Override
    public Bytes setQueryOrd(int ord) throws IOException {
      this.query = vectors1.vectorValue(ord);
      return this;
    }

    @Override
    public float score(int node) throws IOException {
      return similarityFunction.compare(query, vectors2.vectorValue(node));
    }

    @Override
    public Bytes copy() throws IOException {
      return new DefaultRandomVectorScorerSupplier.Bytes(vectors, similarityFunction);
    }
  }
}
