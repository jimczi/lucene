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
import java.util.List;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Bits;

/**
 * Provides random access to vectors by dense ordinal. This interface is used by HNSW-based
 * implementations of KNN search.
 *
 * @lucene.experimental
 */
public interface RandomAccessVectorValues {

  /** Return the number of vector values */
  int size();

  /** Return the dimension of the returned vector values */
  int dimension();

  /**
   * Creates a new copy of this {@link RandomAccessVectorValues}. This is helpful when you need to
   * access different values at once, to avoid overwriting the underlying vector returned.
   */
  RandomAccessVectorValues copy() throws IOException;

  /**
   * Translates vector ordinal to the correct document ID. By default, this is an identity function.
   *
   * @param ord the vector ordinal
   * @return the document Id for that vector ordinal
   */
  default int ordToDoc(int ord) {
    return ord;
  }

  /**
   * Returns the {@link Bits} representing live documents. By default, this is an identity function.
   *
   * @param acceptDocs the accept docs
   * @return the accept docs
   */
  default Bits getAcceptOrds(Bits acceptDocs) {
    return acceptDocs;
  }

  /**
   * Returns the {@link IndexInput} slice that contains the vector values. Since the vectors are
   * stored in a flat file, their offsets are calculable given dimensions and codec configurations.
   * By default, this returns null as some implementations may not reference a slice.
   */
  default IndexInput getSlice() {
    return null;
  }

  /** Float vector values. */
  interface Floats extends RandomAccessVectorValues {
    @Override
    RandomAccessVectorValues.Floats copy() throws IOException;

    /**
     * Return the vector value indexed at the given ordinal.
     *
     * @param targetOrd a valid ordinal, &ge; 0 and &lt; {@link #size()}.
     */
    float[] vectorValue(int targetOrd) throws IOException;
  }

  /** Byte vector values. */
  interface Bytes extends RandomAccessVectorValues {
    @Override
    RandomAccessVectorValues.Bytes copy() throws IOException;

    /**
     * Return the vector value indexed at the given ordinal.
     *
     * @param targetOrd a valid ordinal, &ge; 0 and &lt; {@link #size()}.
     */
    byte[] vectorValue(int targetOrd) throws IOException;
  }

  /**
   * Creates a {@link RandomAccessVectorValues.Floats} from a list of float arrays.
   *
   * @param vectors the list of float arrays
   * @param dim the dimension of the vectors
   * @return a {@link RandomAccessVectorValues.Floats} instance
   */
  static RandomAccessVectorValues.Floats fromFloats(List<float[]> vectors, int dim) {
    return new RandomAccessVectorValues.Floats() {
      @Override
      public int size() {
        return vectors.size();
      }

      @Override
      public int dimension() {
        return dim;
      }

      @Override
      public float[] vectorValue(int targetOrd) throws IOException {
        return vectors.get(targetOrd);
      }

      @Override
      public RandomAccessVectorValues.Floats copy() throws IOException {
        return this;
      }
    };
  }

  /**
   * Creates a {@link RandomAccessVectorValues.Bytes} from a list of byte arrays.
   *
   * @param vectors the list of byte arrays
   * @param dim the dimension of the vectors
   * @return a {@link RandomAccessVectorValues.Bytes} instance
   */
  static RandomAccessVectorValues.Bytes fromBytes(List<byte[]> vectors, int dim) {
    return new RandomAccessVectorValues.Bytes() {
      @Override
      public int size() {
        return vectors.size();
      }

      @Override
      public int dimension() {
        return dim;
      }

      @Override
      public byte[] vectorValue(int targetOrd) throws IOException {
        return vectors.get(targetOrd);
      }

      @Override
      public RandomAccessVectorValues.Bytes copy() throws IOException {
        return this;
      }
    };
  }
}
