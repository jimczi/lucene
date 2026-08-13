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
package org.apache.lucene.sandbox.codecs.perrangepoints;

import java.io.IOException;
import java.util.Arrays;
import org.apache.lucene.index.PointValues;

/**
 * Several ranges' points seen as one field's worth.
 *
 * <p>The ranges are presented as the children of a synthetic root, so a query descends into only
 * the ranges whose bounding box it needs. Below that first level everything delegates to the range's
 * own tree, which is a stock block k-d tree.
 */
final class UnionPointValues extends PointValues {

  private final PointValues[] subs;
  private final byte[] minPacked;
  private final byte[] maxPacked;
  private final long size;
  private final int docCount;

  UnionPointValues(PointValues[] subs) {
    assert subs.length > 0;
    this.subs = subs;
    final int packedIndexBytes = subs[0].getNumIndexDimensions() * subs[0].getBytesPerDimension();
    byte[] min = subs[0].getMinPackedValue().clone();
    byte[] max = subs[0].getMaxPackedValue().clone();
    long totalSize = 0;
    long totalDocs = 0;
    for (PointValues v : subs) {
      widen(min, max, v.getMinPackedValue(), v.getMaxPackedValue(), packedIndexBytes,
          v.getBytesPerDimension());
      totalSize += v.size();
      // A document belongs to exactly one range, so document counts add up rather than overlap.
      totalDocs += v.getDocCount();
    }
    this.minPacked = min;
    this.maxPacked = max;
    this.size = totalSize;
    this.docCount = (int) Math.min(Integer.MAX_VALUE, totalDocs);
  }

  private static void widen(
      byte[] min, byte[] max, byte[] otherMin, byte[] otherMax, int packedIndexBytes, int dimBytes) {
    for (int offset = 0; offset < packedIndexBytes; offset += dimBytes) {
      if (Arrays.compareUnsigned(otherMin, offset, offset + dimBytes, min, offset, offset + dimBytes)
          < 0) {
        System.arraycopy(otherMin, offset, min, offset, dimBytes);
      }
      if (Arrays.compareUnsigned(otherMax, offset, offset + dimBytes, max, offset, offset + dimBytes)
          > 0) {
        System.arraycopy(otherMax, offset, max, offset, dimBytes);
      }
    }
  }

  @Override
  public PointTree getPointTree() throws IOException {
    final PointTree[] roots = new PointTree[subs.length];
    for (int i = 0; i < subs.length; i++) {
      roots[i] = subs[i].getPointTree();
    }
    return new UnionPointTree(roots, minPacked, maxPacked, size);
  }

  @Override
  public byte[] getMinPackedValue() {
    return minPacked;
  }

  @Override
  public byte[] getMaxPackedValue() {
    return maxPacked;
  }

  @Override
  public int getNumDimensions() {
    return subs[0].getNumDimensions();
  }

  @Override
  public int getNumIndexDimensions() {
    return subs[0].getNumIndexDimensions();
  }

  @Override
  public int getBytesPerDimension() {
    return subs[0].getBytesPerDimension();
  }

  @Override
  public long size() {
    return size;
  }

  @Override
  public int getDocCount() {
    return docCount;
  }

  /** A synthetic root whose children are the ranges' own roots. */
  private static final class UnionPointTree implements PointTree {

    private final PointTree[] roots;
    private final PointTree[] live;
    private final byte[] minPacked;
    private final byte[] maxPacked;
    private final long size;

    /** Which range we are inside, or -1 at the synthetic root. */
    private int current = -1;

    /** How far below that range's root we are, so the root level knows to step sideways. */
    private int depth;

    UnionPointTree(PointTree[] roots, byte[] minPacked, byte[] maxPacked, long size) {
      this.roots = roots;
      this.live = new PointTree[roots.length];
      this.minPacked = minPacked;
      this.maxPacked = maxPacked;
      this.size = size;
    }

    /** The range's tree, positioned at its root the first time it is entered. */
    private PointTree enter(int range) {
      if (live[range] == null) {
        live[range] = roots[range].clone();
      }
      return live[range];
    }

    @Override
    public PointTree clone() {
      final PointTree[] clonedRoots = new PointTree[roots.length];
      for (int i = 0; i < roots.length; i++) {
        clonedRoots[i] = roots[i].clone();
      }
      final UnionPointTree copy = new UnionPointTree(clonedRoots, minPacked, maxPacked, size);
      if (current >= 0) {
        // Cloning below the synthetic root clones the range's tree where it stands, so the copy
        // continues from the same place rather than from that range's root.
        copy.current = current;
        copy.depth = depth;
        copy.live[current] = live[current].clone();
      }
      return copy;
    }

    @Override
    public boolean moveToChild() throws IOException {
      if (current < 0) {
        current = 0;
        depth = 0;
        enter(current);
        return true;
      }
      if (enter(current).moveToChild()) {
        depth++;
        return true;
      }
      return false;
    }

    @Override
    public boolean moveToSibling() throws IOException {
      if (current < 0) {
        return false;
      }
      if (depth > 0) {
        return enter(current).moveToSibling();
      }
      if (current + 1 >= roots.length) {
        return false;
      }
      current++;
      enter(current);
      return true;
    }

    @Override
    public boolean moveToParent() throws IOException {
      if (current < 0) {
        return false;
      }
      if (depth > 0) {
        if (enter(current).moveToParent()) {
          depth--;
          return true;
        }
        return false;
      }
      current = -1;
      return true;
    }

    @Override
    public byte[] getMinPackedValue() {
      return current < 0 ? minPacked : enter(current).getMinPackedValue();
    }

    @Override
    public byte[] getMaxPackedValue() {
      return current < 0 ? maxPacked : enter(current).getMaxPackedValue();
    }

    @Override
    public long size() {
      return current < 0 ? size : enter(current).size();
    }

    @Override
    public void visitDocIDs(IntersectVisitor visitor) throws IOException {
      if (current < 0) {
        // A range entered earlier is left wherever it stopped, so visiting everything from the
        // synthetic root has to start each range from its own root again.
        for (PointTree root : roots) {
          root.clone().visitDocIDs(visitor);
        }
      } else {
        enter(current).visitDocIDs(visitor);
      }
    }

    @Override
    public void visitDocValues(IntersectVisitor visitor) throws IOException {
      if (current < 0) {
        for (PointTree root : roots) {
          root.clone().visitDocValues(visitor);
        }
      } else {
        enter(current).visitDocValues(visitor);
      }
    }
  }
}
