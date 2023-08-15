package org.apache.lucene.sandbox.codecs.pq;

import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.NeighborQueue;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.TimeUnit;

public class SIFTBenchmark {
  private static final Path docsPath =
          Paths.get("/Users/jimczi/playground/lucene/benchmarks/datasets/sift/sift_base.fvecs");
  private static final Path trainPath =
          Paths.get("/Users/jimczi/playground/lucene/benchmarks/datasets/sift/sift_learn.fvecs");
  private static final Path queryPath =
          Paths.get("/Users/jimczi/playground/lucene/benchmarks/datasets/sift/sift_query.fvecs");
  private static final int numDocs = 1_000_000;
  private static final int numTraining = 100_000;
  private static final int numQuery = 10_000;
  private static final int numDims = 128;
  private static final int numSubQuantizer = 8;
  private static final int topK = 10;
  private static final int rerankFactor = 10;
  private static final long seed = 42L;

  public static void main(String[] args) throws Exception {
    final ProductQuantizer pq;
    try (FileChannel in = FileChannel.open(trainPath)) {
      long start = System.nanoTime();
      FileFloatVectorValues reader = new FileFloatVectorValues(in, numDims, numTraining);
      pq = ProductQuantizer.create(reader, numSubQuantizer, seed);
      long elapsed = System.nanoTime() - start;
      System.out.format("Create product quantizer using %d training vectors and %d sub-quantizer in %d milliseconds%n",
              numTraining, numSubQuantizer, TimeUnit.NANOSECONDS.toMillis(elapsed));
    }

    final byte[][] codes = new byte[numDocs][];
    try (FileChannel docIn = FileChannel.open(docsPath)) {
      FileFloatVectorValues docReader = new FileFloatVectorValues(docIn, numDims, numDocs);
      long start = System.nanoTime();
      for (int i = 0; i < docReader.size(); i++) {
        codes[i] = pq.encode(docReader.vectorValue(i));
      }
      long elapsed = System.nanoTime() - start;
      System.out.format(Locale.ROOT, "Encode %d vectors in %d milliseconds%n",
              numDocs, numSubQuantizer, TimeUnit.NANOSECONDS.toMillis(elapsed));
    }


    try (FileChannel docIn = FileChannel.open(docsPath);
         FileChannel queryIn = FileChannel.open(queryPath)) {
      FileFloatVectorValues docReader = new FileFloatVectorValues(docIn, numDims, numDocs);
      FileFloatVectorValues queryReader = new FileFloatVectorValues(queryIn, numDims, numQuery);
      int totalMatches = 0;
      int totalResults = 0;
      long totalCodeCompare = 0;
      long totalVectorCompare = 0;
      for (int i = 0; i < 10; i++) {
        float[] candidate = queryReader.vectorValue(i);
        long startCode = System.nanoTime();
        int[] results = getTopDocs(pq, codes, candidate, topK*rerankFactor);
        totalCodeCompare += System.nanoTime() - startCode;

        long startVector = System.nanoTime();
        int[] nn = getNN(docReader, candidate, topK);
        totalVectorCompare = System.nanoTime() - startVector;
        totalMatches += compareNN(nn, results);
        totalResults += nn.length;
      }
      float recall = totalMatches / (float) totalResults;
      System.out.format(Locale.ROOT,
              "Query %d codes in %d milliseconds with a recall@%d of %f with a re-rank factor of %d%n",
              numQuery, TimeUnit.NANOSECONDS.toMillis(totalCodeCompare), topK, recall, rerankFactor);
      System.out.format(Locale.ROOT, "Query %d vectors in %d milliseconds with a recall@%d of 1%n",
              numQuery, TimeUnit.NANOSECONDS.toMillis(totalVectorCompare), topK);
    }
  }

  private static int[] getTopDocs(ProductQuantizer quantizer,
                                  byte[][] codes,
                                  float[] query,
                                  int topK) {
    NeighborQueue pq = new NeighborQueue(topK, true);
    ProductQuantizer.DistanceRunner runner = quantizer.createDistanceRunner(query);
    for (int i = 0; i < codes.length; i++) {
      float res = runner.distance(codes[i]);
      pq.insertWithOverflow(i, res);
    }
    int[] topDocs = new int[topK];
    for (int k = topK - 1; k >= 0; k--) {
      topDocs[k] = pq.topNode();
      pq.pop();
    }
    return topDocs;
  }

  private static int[] getNN(FileFloatVectorValues reader, float[] query, int topK) throws IOException {
    int[] result = new int[topK];
    NeighborQueue queue = new NeighborQueue(topK, true);
    for (int j = 0; j < numDocs; j++) {
      float[] doc = reader.vectorValue(j);
      float d = VectorUtil.squareDistance(query, doc);
      queue.insertWithOverflow(j, d);
    }
    for (int k = topK - 1; k >= 0; k--) {
      result[k] = queue.topNode();
      queue.pop();
    }
    return result;
  }

  private static int compareNN(int[] expected, int[] results) {
    int matched = 0;
    Set<Integer> expectedSet = new HashSet<>();
    for (int i = 0; i < expected.length; i++) {
      expectedSet.add(expected[i]);
    }
    for (int scoreDoc : results) {
      if (expectedSet.contains(scoreDoc)) {
        ++matched;
      }
    }
    return matched;
  }
}
