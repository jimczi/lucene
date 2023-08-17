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

package org.apache.lucene.sandbox.codecs.pq;

import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

import java.io.IOException;

public class ProductQuantizer {
  enum DistanceFunction {
    L2,
    INNER_PRODUCT
  }

  static final int NUM_CENTROIDS = 256;

  private final int numDims;
  private final int numSubQuantizer;
  private final int subVectorLength;
  private final float[][][] centroids;
  private final DistanceFunction distanceFunction;

  private ProductQuantizer(int numDims,
                           int numSubQuantizer,
                           float[][][] centroids,
                           DistanceFunction distanceFunction) {
    this.numDims = numDims;
    this.numSubQuantizer = numSubQuantizer;
    this.subVectorLength = numDims / numSubQuantizer;
    this.centroids = centroids;
    this.distanceFunction = distanceFunction;
  }

  public static ProductQuantizer create(RandomAccessVectorValues<float[]> reader,
                                        int numSubQuantizer,
                                        DistanceFunction distanceFunction,
                                        long seed) throws IOException {
    int subVectorLength = reader.dimension() / numSubQuantizer;
    float[][][] centroids = new float[numSubQuantizer][][];
    for (int i = 0; i < numSubQuantizer; i++) {
      // take the appropriate sub-vector
      int startOffset = i * subVectorLength;
      int endOffset = Math.min(startOffset + subVectorLength, reader.dimension());
      SimpleKMeans kmeans =
              new SimpleKMeans(reader, startOffset, endOffset, NUM_CENTROIDS, seed);
      centroids[i] = kmeans.computeCentroids();
    }
    return new ProductQuantizer(reader.dimension(), numSubQuantizer, centroids, distanceFunction);
  }

  public byte[] encode(float[] vector) {
    byte[] pqCode = new byte[numSubQuantizer];

    for (int i = 0; i < numSubQuantizer; i++) {
      // take the appropriate sub-vector
      int startIndex = i * subVectorLength;
      int endIndex = Math.min(startIndex + subVectorLength, numDims);
      float[] subVector = ArrayUtil.copyOfSubArray(vector, startIndex, endIndex);
      pqCode[i] = computeNearestProductIndex(subVector, i);
    }
    return pqCode;
  }

  public DistanceRunner createDistanceRunner(float[] qVector) {
    float[] distances = new float[numSubQuantizer*NUM_CENTROIDS];
    for (int i = 0; i < numSubQuantizer; i++) {
      // take the appropriate sub-vector
      int startIndex = i * subVectorLength;
      int endIndex = startIndex + subVectorLength;
      float[] subVector = ArrayUtil.copyOfSubArray(qVector, startIndex, endIndex);
      for (int j = 0; j < NUM_CENTROIDS; j++) {
        float dist;
        switch (distanceFunction) {
          case L2 -> dist = 1f - VectorUtil.squareDistance(centroids[i][j], subVector);
          case INNER_PRODUCT -> dist = VectorUtil.dotProduct(centroids[i][j], subVector);
          default -> throw new IllegalArgumentException("not implemented");
        }
        distances[i * NUM_CENTROIDS + j] = dist;
      }
    }
    return new DistanceRunner(distances);
  }

  public QuantizedDistanceRunner createQuantizedDistanceRunner(float[] qVector) {
    DistanceRunner floatRunner = createDistanceRunner(qVector);
    byte[] distanceBytes = new byte[numSubQuantizer * NUM_CENTROIDS];
    for (int i = 0; i < numSubQuantizer; i++) {
      float minDist = Float.POSITIVE_INFINITY;
      float maxDist = Float.NEGATIVE_INFINITY;
      for (int j = 0; j < NUM_CENTROIDS; j++) {
        int index = i * NUM_CENTROIDS + j;
        if (floatRunner.distanceTable[index] < minDist) {
          minDist = floatRunner.distanceTable[index];
        }
        if (floatRunner.distanceTable[index] > maxDist) {
          maxDist = floatRunner.distanceTable[index];
        }
      }
      for (int j = 0; j < NUM_CENTROIDS; j++) {
        int index = i * NUM_CENTROIDS + j;
        distanceBytes[index] = quantize(floatRunner.distanceTable[index], minDist, maxDist);
      }
    }
    return new QuantizedDistanceRunner(distanceBytes);
  }

  public static class DistanceRunner {
    public final float[] distanceTable;

    private DistanceRunner(float[] distances) {
      this.distanceTable = distances;
    }

    public float distance(byte[] pqCode) {
      float score = 0f;
      for (int i = 0; i < pqCode.length; i++) {
        score += distanceTable[i * NUM_CENTROIDS + ((int) pqCode[i] & 0xFF)];
      }
      return score;
    }
  }

  public static class QuantizedDistanceRunner {
    private final byte[] distanceTable;

    private QuantizedDistanceRunner(byte[] distanceTable) {
      this.distanceTable = distanceTable;
    }

    public short distance(byte[] pqCode) {
      short score = 0;
      for (int i = 0; i < pqCode.length; i++) {
        score += distanceTable[i * NUM_CENTROIDS + ((int) pqCode[i] & 0xFF)];
      }
      return score;
    }
  }

  private byte quantizeByte(float sample, float maxValue) {
    return (byte) ((sample / maxValue) * 255.0);
  }

  private byte quantize(float sample, float min, float max){
    return quantizeByte( sample - min, max - min );
  }

  private byte computeNearestProductIndex(float[] subVector, int subIndex) {
    int centroidIndex = -1;
    float bestDistance = Float.NEGATIVE_INFINITY;
    for (int c = 0; c < NUM_CENTROIDS; c++) {
      float dist = 1f - VectorUtil.squareDistance(centroids[subIndex][c], subVector);
      if (dist > bestDistance) {
        bestDistance = dist;
        centroidIndex = c;
      }
    }
    return (byte) centroidIndex;
  }
}
