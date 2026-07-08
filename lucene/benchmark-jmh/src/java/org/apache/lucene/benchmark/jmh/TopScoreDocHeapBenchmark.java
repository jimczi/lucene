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
package org.apache.lucene.benchmark.jmh;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.apache.lucene.util.NumericUtils;
import org.apache.lucene.util.TernaryLongHeap;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Measures the cost of the {@code TopScoreDocCollector} top-N heap when the global doc-id space is
 * widened from {@code int} to {@code long}.
 *
 * <p>The default top-scores collector used to pack {@code (score, doc)} into a single {@code long}
 * in a {@link TernaryLongHeap} (a 32-bit score plus {@code Integer.MAX_VALUE - doc}). A 64-bit doc
 * id no longer fits alongside the score, so the collector now keeps the two components in parallel
 * {@code long[]}/{@code float[]} arrays. This benchmark compares the two under the collector's
 * access pattern:
 *
 * <ul>
 *   <li><b>packedLongHeap</b> — the old {@link TernaryLongHeap} of packed {@code (score, doc)} longs
 *       (only correct for doc ids below {@code Integer.MAX_VALUE}).
 *   <li><b>parallelArrayHeap</b> — the new {@code (long doc, float score)} parallel-array heap.
 * </ul>
 *
 * Docs are fed in increasing doc-id order with random scores; the current bottom score is cached (as
 * in the real collector), so most docs take the cheap non-competitive reject path and only the
 * competitive minority trigger a heap update.
 *
 * <p>The sift-down reads the doc array only to break an exact score tie (rare with real scores; and
 * constant-score queries stop competing once the heap fills), so its hot loop reads a single array
 * most of the time.
 *
 * <p>Representative result (1M docs, Apple M-series, JDK 26, 3 forks; avg time, lower is better):
 *
 * <pre>
 *   topN     packedLongHeap     parallelArrayHeap
 *     10        ~501 us/op          ~492 us/op   (~2% faster)
 *    100        ~327 us/op          ~281 us/op   (~14% faster)
 *   1000        ~470 us/op          ~646 us/op   (~37% slower)
 * </pre>
 *
 * So the parallel-array heap is even-or-faster at the common top-10/top-100 sizes; the residual
 * regression at large top-N is the cost of moving/holding two arrays (a 64-bit doc plus a 32-bit
 * score) rather than one packed long once the heap no longer fits in L1. To avoid it entirely for
 * existing indexes, {@code TopScoreDocCollector} keeps the packed encoding ({@code
 * PackedDocScoreHeap}) when the reader's doc-id space fits {@code int} and uses the parallel-array
 * heap ({@code LongDocScoreHeap}) only for a genuinely &gt; 2^31 space; this benchmark measures the
 * two heaps directly.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(
    value = 3,
    jvmArgsAppend = {"-Xmx1g", "-Xms1g", "-XX:+AlwaysPreTouch"})
public class TopScoreDocHeapBenchmark {

  @Param({"10", "100", "1000"})
  int topN;

  @Param({"1000000"})
  int numDocs;

  private float[] scores;

  @Setup(Level.Trial)
  public void setup() {
    Random r = new Random(0x9E3779B97F4A7C15L);
    scores = new float[numDocs];
    for (int i = 0; i < numDocs; i++) {
      // positive, per the collector contract; a wide range so the competitive fraction is realistic
      scores[i] = 1f + r.nextFloat() * 1000f;
    }
  }

  // ---- old: (score, doc) packed into one long in a TernaryLongHeap ----

  private static long encode(int docId, float score) {
    return (((long) NumericUtils.floatToSortableInt(score)) << 32) | (Integer.MAX_VALUE - docId);
  }

  private static float toScore(long value) {
    return NumericUtils.sortableIntToFloat((int) (value >>> 32));
  }

  private static int docId(long value) {
    return Integer.MAX_VALUE - ((int) value);
  }

  @Benchmark
  public long packedLongHeap() {
    final long least = encode(Integer.MAX_VALUE, Float.NEGATIVE_INFINITY);
    TernaryLongHeap heap = new TernaryLongHeap(topN, least);
    float topScore = toScore(heap.top());
    for (int doc = 0; doc < numDocs; doc++) {
      float score = scores[doc];
      if (score > topScore) {
        long top = heap.updateTop(encode(doc, score));
        topScore = toScore(top);
      }
    }
    long sum = 0;
    for (int i = heap.size(); i > 0; i--) {
      long code = heap.pop();
      sum += docId(code) + Float.floatToIntBits(toScore(code));
    }
    return sum;
  }

  // ---- new: (long doc, float score) parallel-array heap (copy of o.a.l.search.DocScoreHeap) ----

  @Benchmark
  public long parallelArrayHeap() {
    LongDocScoreHeap heap = new LongDocScoreHeap(topN);
    float topScore = heap.topScore();
    for (int doc = 0; doc < numDocs; doc++) {
      float score = scores[doc];
      if (score > topScore) {
        heap.updateTop(doc, score);
        topScore = heap.topScore();
      }
    }
    long sum = 0;
    for (int i = heap.size(); i > 0; i--) {
      sum += heap.topDoc() + Float.floatToIntBits(heap.topScore());
      heap.pop();
    }
    return sum;
  }

  /** Faithful copy of {@code org.apache.lucene.search.DocScoreHeap} (which is package-private). */
  private static final class LongDocScoreHeap {
    private static final int ARITY = 3;
    private final long[] docs;
    private final float[] scores;
    private int size;

    LongDocScoreHeap(int capacity) {
      if (capacity < 1) {
        capacity = 1;
      }
      docs = new long[capacity + 1];
      scores = new float[capacity + 1];
      Arrays.fill(docs, 1, capacity + 1, Long.MAX_VALUE);
      Arrays.fill(scores, 1, capacity + 1, Float.NEGATIVE_INFINITY);
      size = capacity;
    }

    long topDoc() {
      return docs[1];
    }

    float topScore() {
      return scores[1];
    }

    int size() {
      return size;
    }

    void updateTop(long doc, float score) {
      docs[1] = doc;
      scores[1] = score;
      downHeap(1);
    }

    void pop() {
      docs[1] = docs[size];
      scores[1] = scores[size];
      size--;
      downHeap(1);
    }

    private void downHeap(int i) {
      final long doc = docs[i];
      final float score = scores[i];
      for (; ; ) {
        int firstChild = ARITY * (i - 1) + 2;
        if (firstChild > size) {
          break;
        }
        int lastChild = Math.min(firstChild + ARITY - 1, size);
        int best = firstChild;
        float bestScore = scores[firstChild];
        for (int c = firstChild + 1; c <= lastChild; c++) {
          float childScore = scores[c];
          if (childScore < bestScore || (childScore == bestScore && docs[c] > docs[best])) {
            best = c;
            bestScore = childScore;
          }
        }
        if (score < bestScore || (score == bestScore && doc > docs[best])) {
          break;
        }
        docs[i] = docs[best];
        scores[i] = bestScore;
        i = best;
      }
      docs[i] = doc;
      scores[i] = score;
    }
  }
}
