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
package org.apache.lucene.sandbox.document;

import org.apache.lucene.document.BinaryDocValuesField;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.QueryVisitor;
import org.apache.lucene.search.ScoreMode;
import org.apache.lucene.search.Scorer;
import org.apache.lucene.search.ScorerSupplier;
import org.apache.lucene.search.Weight;
import org.apache.lucene.util.BytesRef;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class LateInteractionField extends BinaryDocValuesField {
  private final int numDims;

  public LateInteractionField(String fieldName, float[][] matrix, int numDims) {
    super(fieldName, encode(matrix, numDims));
    this.numDims = numDims;
  }

  public void setMatrixValue(float[][] matrix) {
    this.fieldsData = encode(matrix, numDims);
  }

  public Query newSlowLateInteractionQuery(float[][] vectors, VectorSimilarityFunction similarityFunction) {
    return new LateInteractionQuery(name(), vectors, similarityFunction, numDims);
  }

  private static BytesRef encode(float[][] matrix, int expectedNumDims) {
    final ByteBuffer buffer = ByteBuffer.allocate(matrix.length * expectedNumDims * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putInt(matrix[0].length);
    for (float[] v : matrix) {
      if (v.length != expectedNumDims) {
        throw new IllegalStateException("");
      }
      buffer.asFloatBuffer().put(v);
    }
    return new BytesRef(buffer.array());
  }

  private static float[][] decode(BytesRef payload, int expectedNumDims) {
    final ByteBuffer buffer = ByteBuffer.wrap(payload.bytes, payload.offset, payload.length);
    int numDims = buffer.getInt();
    if (expectedNumDims != numDims) {
      throw new IllegalStateException("");
    }
    var floatBuffer = buffer.asFloatBuffer();
    int numVectors = (payload.length - 4) / numDims;
    float[][] res = new float[numVectors][];
    for (int i = 0; i < numVectors; i++) {
      res[i] = new float[numDims];
      floatBuffer.get(res[i]);
    }
    return res;
  }

  private static class LateInteractionQuery extends Query {
    private final String fieldName;
    private final float[][] queryVectors;
    private final VectorSimilarityFunction similarityFunction;
    private final int expectedDimSize;

    private LateInteractionQuery(String fieldName, float[][] queryVectors, VectorSimilarityFunction similarityFunction, int expectedDimSize) {
      this.fieldName = fieldName;
      this.queryVectors = queryVectors;
      this.similarityFunction = similarityFunction;
      this.expectedDimSize = expectedDimSize;
    }

    @Override
    public Weight createWeight(IndexSearcher searcher, ScoreMode scoreMode, float boost) throws IOException {
      return new Weight(this) {
        @Override
        public Explanation explain(LeafReaderContext context, int doc) throws IOException {
          // TODO
          return null;
        }

        @Override
        public ScorerSupplier scorerSupplier(LeafReaderContext context) throws IOException {
          var values = context.reader().getBinaryDocValues(fieldName);
          if (values == null) {
            return null;
          }
          return new ScorerSupplier() {
            @Override
            public Scorer get(long leadCost) {
              return new Scorer() {
                @Override
                public int docID() {
                  return values.docID();
                }

                @Override
                public DocIdSetIterator iterator() {
                  return values;
                }

                @Override
                public float getMaxScore(int upTo) {
                  return Float.MAX_VALUE;
                }

                @Override
                public float score() throws IOException {
                  float[][] docVectors = decode(values.binaryValue(), expectedDimSize);
                  float result = 0;
                  for (float[] o : queryVectors) {
                    float maxSim = Float.MIN_VALUE;
                    for (float[] i : docVectors) {
                      maxSim = Float.max(maxSim, similarityFunction.compare(o, i));
                    }
                    result += maxSim;
                  }
                  return result;
                }
              };
            }

            @Override
            public long cost() {
              return values.cost();
            }
          };
        }

        @Override
        public boolean isCacheable(LeafReaderContext ctx) {
          return false;
        }
      };
    }

    @Override
    public String toString(String field) {
      return "LateInteractionQuery";
    }

    @Override
    public void visit(QueryVisitor visitor) {
      if (visitor.acceptField(fieldName)) {
        visitor.visitLeaf(this);
      }
    }

    @Override
    public boolean equals(Object obj) {
      // TODO
      return false;
    }

    @Override
    public int hashCode() {
      // TODO
      return 0;
    }
  }
}
