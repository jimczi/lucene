package org.apache.lucene.sandbox.codecs.pq;

import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.NeighborQueue;
import org.apache.lucene.sandbox.codecs.pq.ProductQuantizer.DistanceFunction;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.TimeUnit;

public class Benchmark {
  private static final Path dirPath =
          Paths.get("/Users/jimczi/playground/lucene/benchmarks/datasets/wikipedia-22-12-en-embeddings");
  private static final String vectorFile = "base.fvecs";
  private static final String trainFile = "base.fvecs";
  private static final String queryFile = "base.fvecs";
  private static final int numDocs = 1000000;
  private static final int numTraining = 10000;
  private static final int numQuery = 100;
  private static final int numDims = 768;
  private static DistanceFunction distanceFunction = DistanceFunction.INNER_PRODUCT;
  private static int numSubQuantizer = numDims / 16;
  private static int topK = 10;
  private static int rerankFactor = 100;
  private static boolean sphericalKMeans = false;
  private static long seed = 42L;

  public static void main(String[] args) throws Exception {
    for (int i = 0; i < args.length; i++) {
      String arg = args[i];
      switch (arg) {
        case "-numSub":
          numSubQuantizer = Integer.parseInt(args[++i]);
          break;
        case "-topk":
          topK = Integer.parseInt(args[++i]);
          break;
        case "-rerankFactor":
          rerankFactor = Integer.parseInt(args[++i]);
          break;
        case "-spherical":
          sphericalKMeans = true;
          break;
        case "-seed":
          seed = Long.parseLong(args[++i]);
          break;
        case "-metric":
          String metric = args[++i];
          switch (metric) {
            case "l2":
              distanceFunction = DistanceFunction.L2;
              break;
            case "ip":
              distanceFunction = DistanceFunction.INNER_PRODUCT;
              break;
            default:
              usage();
              throw new IllegalArgumentException("-metric can be 'le' or 'ip' only");
          }
          break;
        default:
          usage();
          throw new IllegalArgumentException("unknown argument " + arg);
      }
    }
    new Benchmark().runBenchmark();
  }

  private static void usage() {
    String error =
            "Usage: Benchmark [-numSub N] [-spherical] [-metric N] [-topk N] [-rerankFactor N] [-seed N]";
    System.err.println(error);
  }

  private void runBenchmark() throws Exception {
    final ProductQuantizer pq;
    try (MMapDirectory directory = new MMapDirectory(dirPath);
         IndexInputFloatVectorValues trainingInput =
              new IndexInputFloatVectorValues(directory, trainFile, numDims, numTraining);
         IndexInputFloatVectorValues vectorInput =
              new IndexInputFloatVectorValues(directory, vectorFile, numDims, numDocs);
         IndexInputFloatVectorValues queryInput =
              new IndexInputFloatVectorValues(directory, queryFile, numDims, numQuery)) {
      long start = System.nanoTime();
      pq = ProductQuantizer.create(trainingInput, numSubQuantizer, sphericalKMeans, seed);
      long elapsed = System.nanoTime() - start;
      System.out.format("Create product quantizer using %d training vectors and %d sub-quantizers in %d milliseconds%n",
              numTraining, numSubQuantizer, TimeUnit.NANOSECONDS.toMillis(elapsed));

      final byte[][] codes = new byte[numDocs][];
      long startEncode = System.nanoTime();
      for (int i = 0; i < vectorInput.size(); i++) {
        codes[i] = pq.encode(vectorInput.vectorValue(i));
      }
      long elapsedEncode = System.nanoTime() - startEncode;
      System.out.format(Locale.ROOT, "Encode %d vectors with %d sub-quantizers in %d milliseconds%n",
              numDocs, numSubQuantizer, TimeUnit.NANOSECONDS.toMillis(elapsedEncode));

      int totalMatches = 0;
      int totalResults = 0;
      long totalCodeCompare = 0;
      long totalVectorCompare = 0;
      for (int i = 0; i < numQuery; i++) {
        float[] candidate = queryInput.vectorValue(i);
        long startCode = System.nanoTime();
        int[] results = getTopDocs(pq, distanceFunction, codes, candidate, topK * rerankFactor);
        totalCodeCompare += System.nanoTime() - startCode;

        long startVector = System.nanoTime();
        int[] nn = getNN(vectorInput, candidate, topK);
        totalVectorCompare += System.nanoTime() - startVector;
        totalMatches += compareNN(nn, results);
        totalResults += nn.length;
      }
      float recall = totalMatches / (float) totalResults;
      System.out.format(Locale.ROOT,
              "Query %d codes in %d milliseconds with a recall@%d of %f with a re-rank factor of %d%n",
              numQuery, TimeUnit.NANOSECONDS.toMillis(totalCodeCompare), topK, recall, rerankFactor);
      System.out.format(Locale.ROOT, "Query %d vectors in %d milliseconds with a recall@%d of %f%n",
              numQuery, TimeUnit.NANOSECONDS.toMillis(totalVectorCompare), topK, 1.0f);
    }
  }

  private int[] getTopDocs(ProductQuantizer quantizer,
                           DistanceFunction distanceFunction,
                           byte[][] codes,
                           float[] query,
                           int topK) {
    NeighborQueue pq = new NeighborQueue(topK, false);
    ProductQuantizer.DistanceRunner runner = quantizer.createDistanceRunner(query, distanceFunction);
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

  private int[] getNN(IndexInputFloatVectorValues reader,
                      float[] query,
                      int topK) throws IOException {
    int[] result = new int[topK];
    NeighborQueue queue = new NeighborQueue(topK, false);
    for (int j = 0; j < numDocs; j++) {
      float[] doc = reader.vectorValue(j);
      float dist;
      switch (distanceFunction) {
        case L2 -> dist = 1f - VectorUtil.squareDistance(query, doc);
        case INNER_PRODUCT -> dist = VectorUtil.dotProduct(query, doc);
        default -> throw new IllegalArgumentException("Not implemented");
      }
      queue.insertWithOverflow(j, dist);
    }
    for (int k = topK - 1; k >= 0; k--) {
      result[k] = queue.topNode();
      queue.pop();
    }
    return result;
  }

  private int compareNN(int[] expected, int[] results) {
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
