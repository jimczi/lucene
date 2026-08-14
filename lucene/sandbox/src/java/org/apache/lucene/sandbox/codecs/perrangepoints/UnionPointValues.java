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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.apache.lucene.index.PointValues;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.BytesRef;
import org.apache.lucene.util.PriorityQueue;

/**
 * Several ranges' points seen as one field's worth.
 *
 * <p>How they are presented depends on how many data dimensions the field has, because only one of
 * the two cases carries an ordering contract. With more than one, the ranges become the children of
 * a synthetic root and a query descends into just the ones whose bounding box it needs. With
 * exactly one, a full traversal owes the caller a single ascending sweep of the values, which no
 * arrangement of per-range children can give -- so the ranges are interleaved point by point
 * instead. See {@link MergedPointTree}.
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
      widen(
          min,
          max,
          v.getMinPackedValue(),
          v.getMaxPackedValue(),
          packedIndexBytes,
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
      byte[] min,
      byte[] max,
      byte[] otherMin,
      byte[] otherMax,
      int packedIndexBytes,
      int dimBytes) {
    for (int offset = 0; offset < packedIndexBytes; offset += dimBytes) {
      if (Arrays.compareUnsigned(
              otherMin, offset, offset + dimBytes, min, offset, offset + dimBytes)
          < 0) {
        System.arraycopy(otherMin, offset, min, offset, dimBytes);
      }
      if (Arrays.compareUnsigned(
              otherMax, offset, offset + dimBytes, max, offset, offset + dimBytes)
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
    if (getNumDimensions() == 1) {
      // One data dimension carries an ordering contract -- a full traversal must sweep the values
      // once, ascending, tie-broken by document id -- and no arrangement of per-range children can
      // satisfy it, since the ranges each span the whole value space. So the ranges are interleaved
      // point by point instead. See MergedPointTree.
      return new MergedPointTree(
          roots, minPacked, maxPacked, size, getNumDimensions() * getBytesPerDimension());
    }
    // More than one data dimension has no such contract: leaves are ordered by whichever dimension
    // compresses best, not by value. The ranges can stay a level of the tree, which keeps a query's
    // pruning hierarchical and lets a cell that is entirely inside it skip reading values at all.
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

  /**
   * The ranges interleaved into one ascending sweep, for the single-data-dimension case.
   *
   * <p>A full traversal of one data dimension must visit values in ascending order, tie-broken by
   * increasing document id. Presenting the ranges as a level of the tree cannot do that: {@link
   * PointValues#intersect} walks children in turn, so it would emit all of range 0's values and
   * then all of range 1's, and the ranges each span the whole value space. The interleaving has to
   * happen point by point, which is what this does -- a k-way merge over one cursor per range.
   *
   * <p>So this tree is flat, and the merge happens in {@link #visitDocValues}. Pruning is not lost
   * with the hierarchy: each cursor still walks its own range's real tree and still asks the
   * visitor about every cell, so a selective query skips exactly the cells it would have skipped
   * anyway. What it does cost is that a cell lying entirely inside the query can no longer skip
   * reading its values -- the merge needs them to order by -- plus a heap operation per point.
   *
   * <p>Both costs fall on whole-index queries only. A query for a single range reads that range's
   * points directly, which is an ordinary block k-d tree with nothing added.
   */
  private static final class MergedPointTree implements PointTree {

    private final PointTree[] roots;
    private final byte[] minPacked;
    private final byte[] maxPacked;
    private final long size;
    private final int packedBytes;

    MergedPointTree(
        PointTree[] roots, byte[] minPacked, byte[] maxPacked, long size, int packedBytes) {
      this.roots = roots;
      this.minPacked = minPacked;
      this.maxPacked = maxPacked;
      this.size = size;
      this.packedBytes = packedBytes;
    }

    @Override
    public PointTree clone() {
      final PointTree[] copies = new PointTree[roots.length];
      for (int i = 0; i < roots.length; i++) {
        copies[i] = roots[i].clone();
      }
      // Flat and stateless: a clone is always another view of the whole thing, never a position
      // part way through it.
      return new MergedPointTree(copies, minPacked, maxPacked, size, packedBytes);
    }

    @Override
    public boolean moveToChild() {
      return false;
    }

    @Override
    public boolean moveToSibling() {
      return false;
    }

    @Override
    public boolean moveToParent() {
      return false;
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
    public long size() {
      return size;
    }

    @Override
    public void visitDocIDs(IntersectVisitor visitor) throws IOException {
      // Ranges are intervals of document id in increasing order, so taking them in turn already
      // yields ascending document ids and nothing has to be merged.
      for (PointTree root : roots) {
        root.clone().visitDocIDs(visitor);
      }
    }

    @Override
    public void visitDocValues(IntersectVisitor visitor) throws IOException {
      final List<RangeCursor> started = new ArrayList<>(roots.length);
      for (PointTree root : roots) {
        final RangeCursor cursor = new RangeCursor(root.clone(), visitor, packedBytes);
        if (cursor.next()) {
          started.add(cursor);
        }
      }
      if (started.isEmpty()) {
        return;
      }
      // A hint about capacity, so a visitor that builds a doc id set sizes itself once. Only the
      // points that survive each cursor's own pruning are actually visited.
      visitor.grow((int) Math.min(Integer.MAX_VALUE, size));

      if (started.size() == 1) {
        final RangeCursor only = started.get(0);
        final Announcer announcer = new Announcer(visitor);
        do {
          announcer.announce(only);
          visitor.visit(only.docID(), only.value());
        } while (only.next());
        return;
      }

      final PriorityQueue<RangeCursor> queue =
          PriorityQueue.usingLessThan(
              started.size(),
              (a, b) -> {
                final int cmp =
                    Arrays.compareUnsigned(a.value(), 0, packedBytes, b.value(), 0, packedBytes);
                // Equal values are ordered by document id, which the contract also asks for. The
                // ranges are disjoint document intervals, so this only ever compares across them.
                return cmp != 0 ? cmp < 0 : a.docID() < b.docID();
              });
      for (RangeCursor cursor : started) {
        queue.add(cursor);
      }
      final Announcer announcer = new Announcer(visitor);
      while (queue.size() > 0) {
        final RangeCursor top = queue.top();
        announcer.announce(top);
        visitor.visit(top.docID(), top.value());
        if (top.next()) {
          queue.updateTop();
        } else {
          queue.pop();
        }
      }
    }
  }

  /**
   * Keeps the cell a visitor was last told about in step with the point about to be handed to it.
   *
   * <p>A visitor is entitled to assume that a point it is shown lies inside the cell of the most
   * recent {@link IntersectVisitor#compare} -- that is how the two calls are paired everywhere
   * else. Interleaving ranges breaks that pairing on its own: a cursor asks about its own cell
   * while looking for its next leaf, and the point emitted next may well come from a different
   * range whose values are nowhere near that cell. So whenever the merge starts drawing from a
   * different leaf, that leaf's bounds are announced first, and every point that follows is inside
   * them.
   */
  private static final class Announcer {

    private final IntersectVisitor visitor;
    private RangeCursor announced;
    private long generation = -1;

    Announcer(IntersectVisitor visitor) {
      this.visitor = visitor;
    }

    void announce(RangeCursor cursor) {
      if (cursor == announced && cursor.leafGeneration() == generation) {
        return;
      }
      announced = cursor;
      generation = cursor.leafGeneration();
      visitor.compare(cursor.cellMin(), cursor.cellMax());
    }
  }

  /**
   * One range's points, pulled one at a time in ascending order, skipping whatever the visitor says
   * it does not want.
   *
   * <p>A block k-d tree over one data dimension is ordered by value, both in how its cells nest and
   * within a leaf, so walking it depth first and reading each leaf in turn already yields ascending
   * values. This buffers a leaf at a time and hands the points back one by one.
   */
  private static final class RangeCursor {

    private final PointTree tree;
    private final IntersectVisitor visitor;
    private final LeafBuffer buffer;
    private byte[] cellMin;
    private byte[] cellMax;
    private long leafGeneration;
    private boolean exhausted;
    private int upto;

    RangeCursor(PointTree tree, IntersectVisitor visitor, int packedBytes) {
      this.tree = tree;
      this.visitor = visitor;
      this.buffer = new LeafBuffer(packedBytes);
    }

    int docID() {
      return buffer.docs[upto];
    }

    byte[] value() {
      return buffer.valueAt(upto);
    }

    /** Bounds of the leaf currently buffered, which contain every point it holds. */
    byte[] cellMin() {
      return cellMin;
    }

    byte[] cellMax() {
      return cellMax;
    }

    /** Changes whenever a new leaf is buffered, so a repeat announcement can be skipped. */
    long leafGeneration() {
      return leafGeneration;
    }

    /** Moves to the next point, returning false once this range has no more. */
    boolean next() throws IOException {
      // The buffer comes first: filling it can reach the end of the tree at the same time, and
      // those buffered points still have to be handed out before the cursor is done.
      if (upto + 1 < buffer.count) {
        upto++;
        return true;
      }
      if (exhausted) {
        return false;
      }
      return fillNextLeaf();
    }

    private boolean fillNextLeaf() throws IOException {
      while (exhausted == false) {
        boolean filled = false;
        if (visitor.compare(tree.getMinPackedValue(), tree.getMaxPackedValue())
            != Relation.CELL_OUTSIDE_QUERY) {
          if (tree.moveToChild()) {
            continue; // descend, and judge the child on its own bounds
          }
          // Captured before stepping away: these are what the points just buffered live inside,
          // and the tree will be sitting on a different cell by the time they are handed out.
          cellMin = tree.getMinPackedValue().clone();
          cellMax = tree.getMaxPackedValue().clone();
          buffer.reset();
          tree.visitDocValues(buffer);
          filled = buffer.count > 0;
        }
        // Step past the cell just handled, whether it was read or skipped.
        while (tree.moveToSibling() == false) {
          if (tree.moveToParent() == false) {
            exhausted = true;
            break;
          }
        }
        if (filled) {
          upto = 0;
          leafGeneration++;
          return true;
        }
      }
      return false;
    }
  }

  /** Collects one leaf's points so they can be handed out one at a time. */
  private static final class LeafBuffer implements IntersectVisitor {

    private final int packedBytes;
    private final byte[] scratch;
    private byte[] values = BytesRef.EMPTY_BYTES;
    private int[] docs = new int[0];
    private int count;

    LeafBuffer(int packedBytes) {
      this.packedBytes = packedBytes;
      this.scratch = new byte[packedBytes];
    }

    void reset() {
      count = 0;
    }

    byte[] valueAt(int index) {
      System.arraycopy(values, index * packedBytes, scratch, 0, packedBytes);
      return scratch;
    }

    @Override
    public void grow(int expected) {
      if (docs.length < count + expected) {
        docs = ArrayUtil.grow(docs, count + expected);
        values = ArrayUtil.grow(values, (count + expected) * packedBytes);
      }
    }

    @Override
    public void visit(int docID) {
      throw new UnsupportedOperationException("a leaf is read for its values, not only its docs");
    }

    @Override
    public void visit(int docID, byte[] packedValue) {
      if (count == docs.length) {
        docs = ArrayUtil.grow(docs, count + 1);
        values = ArrayUtil.grow(values, (count + 1) * packedBytes);
      }
      System.arraycopy(packedValue, 0, values, count * packedBytes, packedBytes);
      docs[count++] = docID;
    }

    @Override
    public Relation compare(byte[] minPackedValue, byte[] maxPackedValue) {
      // Everything reaching here has already been judged against the real visitor by the cursor;
      // saying the cell crosses keeps the reader on the path that hands back values.
      return Relation.CELL_CROSSES_QUERY;
    }
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
