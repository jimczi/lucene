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

package org.apache.lucene.codecs.bitvectors;

import java.io.IOException;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

/** A bit vector scorer for scoring byte vectors. */
public class BitRandomVectorScorerSupplier implements RandomVectorScorerSupplier {
  @Override
  public RandomVectorScorer create(
      RandomAccessVectorValues values, VectorSimilarityFunction similarityFunction)
      throws IOException {
    if (values instanceof RandomAccessVectorValues.Bytes bytes) {
      return new BitRandomVectorScorer(bytes, null);
    } else {
      throw new IllegalArgumentException("");
    }
  }

  @Override
  public RandomVectorScorer create(
      RandomAccessVectorValues values, VectorSimilarityFunction similarityFunction, byte[] query)
      throws IOException {
    if (values instanceof RandomAccessVectorValues.Bytes bytes) {
      return new BitRandomVectorScorer(bytes, query);
    } else {
      throw new IllegalArgumentException("");
    }
  }

  @Override
  public RandomVectorScorer create(
      RandomAccessVectorValues values, VectorSimilarityFunction similarityFunction, float[] query)
      throws IOException {
    throw new IllegalArgumentException("");
  }

  static class BitRandomVectorScorer extends RandomVectorScorer {
    private final RandomAccessVectorValues.Bytes vectorValues;
    private final RandomAccessVectorValues.Bytes vectorValues1;
    private final RandomAccessVectorValues.Bytes vectorValues2;
    private final int bitDimensions;
    private byte[] query;

    private BitRandomVectorScorer(RandomAccessVectorValues.Bytes vectorValues, byte[] query)
        throws IOException {
      super(vectorValues);
      this.vectorValues = vectorValues;
      this.vectorValues1 = vectorValues.copy();
      this.vectorValues2 = vectorValues.copy();
      this.bitDimensions = vectorValues.dimension() * Byte.SIZE;
      this.query = query;
    }

    @Override
    public BitRandomVectorScorer copy() throws IOException {
      return new BitRandomVectorScorer(vectorValues.copy(), null);
    }

    @Override
    public BitRandomVectorScorer setQueryOrd(int ord) throws IOException {
      this.query = vectorValues1.vectorValue(ord);
      return this;
    }

    @Override
    public float score(int node) throws IOException {
      return (bitDimensions - VectorUtil.xorBitCount(query, vectorValues2.vectorValue(node)))
          / (float) bitDimensions;
    }
  }

  @Override
  public String toString() {
    return "BitRandomVectorScorerSupplier()";
  }
}
