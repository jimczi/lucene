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
package org.apache.lucene.index;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.lucene.codecs.FieldsConsumer;
import org.apache.lucene.codecs.FieldsProducer;
import org.apache.lucene.codecs.NormsProducer;
import org.apache.lucene.codecs.TermsPushWriter;
import org.apache.lucene.util.BytesRef;

/**
 * Merges an inverted index into several outputs in a <b>single pass</b> over the inputs.
 *
 * <p>Running an ordinary merge once per output costs {@code k} full reads of the terms dictionary
 * and of every posting, because a terms dictionary is ordered by term rather than by document.
 * Stored fields, doc values, norms and term vectors are keyed by document, so an output's documents
 * are a contiguous interval its reader seeks past and masking the rest genuinely saves the IO. A
 * term's postings are spread across every output instead, so masking documents saves nothing at
 * all. Measured on a text corpus at 64 outputs, a merge that writes its outputs one at a time reads
 * 9.59 times what it writes, against 2.62 for one that shares this pass -- so this is the dominant
 * per-output cost of a partitioned merge, and the reason the API below exists.
 *
 * <p>Points are ordered by value as well, and so have the same problem -- but not the same remedy.
 * A block k-d tree shares nothing between the documents of different outputs the way a dictionary
 * shares one entry for a term, so splitting it costs little and the answer there is to store the
 * points per range to begin with, rather than to divide one tree k ways while merging.
 *
 * <p>The observation that removes it: each output owns a <b>contiguous, increasing</b> interval of
 * the merged document space, and postings arrive in increasing document order. So one term's
 * postings decompose into consecutive runs, one per output, and a single shared cursor can feed
 * every output in turn without buffering anything. Each posting is read exactly once.
 *
 * <p>This requires the outputs to partition the merged document space in order, which holds when
 * the merge is partitioned by an index sort -- the outputs are then key ranges, and the merged
 * order is key order. Callers must not use this otherwise.
 *
 * @lucene.experimental
 */
final class MultiOutputTermsMerger {

  private MultiOutputTermsMerger() {}

  /**
   * Writes the merged postings of {@code mergeState} into {@code consumers}, one per output.
   *
   * <p>Callers decide whether this path is available by asking the postings <i>format</i> (see
   * {@link org.apache.lucene.codecs.PostingsFormat#supportsPushWriter(FieldInfos)}) before creating
   * any consumer: a consumer that cannot be pushed to is discovered here only after files exist.
   *
   * @param mergeState the whole merged document space at once -- see {@link #wholeMergedSpace}
   * @param consumers one consumer per output, in increasing document order
   * @param fieldInfos each output's own {@link FieldInfos}; a per-field dispatching format records
   *     which format wrote a field on the {@link FieldInfo} it is handed, and it is the output's
   *     own one that gets persisted
   * @param norms each output's norms, read back after being written: the postings writer derives
   *     impacts from them and looks them up by that output's document ids
   * @param outputStarts {@code consumers.length + 1} boundaries in merged document space; output
   *     {@code o} owns {@code [outputStarts[o], outputStarts[o+1])}
   */
  static void merge(
      MergeState mergeState,
      FieldsConsumer[] consumers,
      FieldInfos[] fieldInfos,
      NormsProducer[] norms,
      int[] outputStarts)
      throws IOException {
    assert outputStarts.length == consumers.length + 1;
    assert fieldInfos.length == consumers.length;
    assert norms.length == consumers.length;

    final List<Fields> fields = new ArrayList<>();
    final List<ReaderSlice> slices = new ArrayList<>();
    int docBase = 0;
    for (int readerIndex = 0; readerIndex < mergeState.fieldsProducers.length; readerIndex++) {
      final FieldsProducer f = mergeState.fieldsProducers[readerIndex];
      final int maxDoc = mergeState.maxDocs[readerIndex];
      if (f != null) {
        // No checkIntegrity here, unlike FieldsConsumer.merge: the caller of a partitioned merge
        // verifies every input once, before any output exists.
        mergeState.checkAborted();
        slices.add(new ReaderSlice(docBase, maxDoc, readerIndex));
        fields.add(f);
      }
      docBase += maxDoc;
    }

    final Fields merged =
        new MappedMultiFields(
            mergeState,
            new MultiFields(fields.toArray(Fields[]::new), slices.toArray(ReaderSlice[]::new)));

    for (String field : merged) {
      final Terms terms = merged.terms(field);
      if (terms == null) {
        continue;
      }

      final TermsPushWriter[] writers = new TermsPushWriter[consumers.length];
      try {
        for (int o = 0; o < consumers.length; o++) {
          final FieldInfo fieldInfo = fieldInfos[o].fieldInfo(field);
          writers[o] = consumers[o].pushWriter(fieldInfo);
          if (writers[o] == null) {
            // supportsPushWriter() promised otherwise, and by now earlier fields may already be
            // written, so there is no going back to the per-output path. A codec that mixes
            // push-capable and push-incapable formats has to report false for the whole consumer.
            throw new IllegalStateException(
                "supportsPushWriter() returned true but pushWriter() returned null for field '"
                    + field
                    + "' on "
                    + consumers[o].getClass().getName());
          }
        }

        final TermsEnum termsEnum = terms.iterator();
        final SplitPostings split = new SplitPostings(outputStarts);
        BytesRef term;
        while ((term = termsEnum.next()) != null) {
          mergeState.checkAborted();
          split.reset(termsEnum);
          for (int o = 0; o < consumers.length; o++) {
            // An output with no document for this term yields an immediately-exhausted enum, so
            // the postings writer reports a null term state and the term is skipped for it. That
            // is the same contract write(Fields) relies on.
            writers[o].write(term, split.viewFor(o), norms[o]);
          }
        }
      } finally {
        for (TermsPushWriter w : writers) {
          if (w != null) {
            w.close();
          }
        }
      }
    }
  }

  /**
   * The whole merged document space of a partitioned merge, as one {@link MergeState}.
   *
   * <p>Only what a postings merge reads is meaningful on the result: the postings producers, the
   * per-reader document counts, and document maps into the <b>whole</b> merged space rather than
   * into one output. The rest is carried over from the first output and must not be relied on.
   *
   * <p>Those document maps are composed from the per-output ones rather than derived afresh from
   * the unmasked readers. Composing is both cheaper -- it reuses a sort the outputs already paid
   * for -- and exact: an output's document ids <i>are</i> what its own map produced, so offsetting
   * each map by where its output starts can only agree with what the outputs actually contain,
   * whereas a separately derived map would merely be expected to.
   *
   * @param outputs each output's merge state, including outputs that ended up empty
   * @param partitions per input reader, the {@code k+1} document boundaries that split it
   * @param outputStarts where each output begins in the merged space, {@code k+1} entries
   */
  static MergeState wholeMergedSpace(
      List<MergeState> outputs, int[][] partitions, int[] outputStarts) {
    final MergeState first = outputs.get(0);
    final MergeState.DocMap[] docMaps = new MergeState.DocMap[first.docMaps.length];
    for (int i = 0; i < docMaps.length; i++) {
      final int reader = i;
      final int[] bounds = partitions[i];
      docMaps[i] =
          docID -> {
            final int output = outputOf(bounds, docID);
            final int mapped = outputs.get(output).docMaps[reader].get(docID);
            return mapped < 0 ? -1 : outputStarts[output] + mapped;
          };
    }
    // Never the sequential DocIDMerger: a reader contributes to every output, so its documents do
    // not form one run of the merged space no matter how the outputs are ordered.
    return new MergeState(first, docMaps, true);
  }

  /** The output owning {@code doc}, i.e. the only {@code o} with {@code b[o] <= doc < b[o+1]}. */
  private static int outputOf(int[] bounds, int doc) {
    int lo = 0;
    int hi = bounds.length - 2;
    while (lo < hi) {
      final int mid = (lo + hi) >>> 1;
      if (bounds[mid + 1] > doc) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    return lo;
  }

  /**
   * One term's postings, handed out as consecutive per-output views over a single shared cursor.
   * Doc ids are rebased into each output's own document space.
   */
  private static final class SplitPostings {
    private final int[] outputStarts;
    private TermsEnum source;
    private PostingsEnum shared;
    private int current = -1;

    SplitPostings(int[] outputStarts) {
      this.outputStarts = outputStarts;
    }

    void reset(TermsEnum source) {
      this.source = source;
      this.shared = null; // created lazily, with the flags the postings writer asks for
      this.current = -1;
    }

    TermsEnum viewFor(int output) {
      return new FilterLeafReader.FilterTermsEnum(source) {
        @Override
        public PostingsEnum postings(PostingsEnum reuse, int flags) throws IOException {
          if (shared == null) {
            shared = source.postings(null, flags);
            current = shared.nextDoc();
          }
          return new BoundedPostings(output);
        }
      };
    }

    /**
     * The run of the shared cursor that falls inside one output. The cursor is advanced only once
     * this output has consumed its current document, so when the run ends the cursor is left
     * sitting on the first document of the NEXT output rather than past it.
     */
    private final class BoundedPostings extends PostingsEnum {
      private final int lo;
      private final int hi;
      private boolean primed;
      private boolean exhausted;
      private int doc = -1;

      BoundedPostings(int output) {
        this.lo = outputStarts[output];
        this.hi = outputStarts[output + 1];
      }

      @Override
      public int docID() {
        return doc;
      }

      @Override
      public int nextDoc() throws IOException {
        if (exhausted) {
          return doc = NO_MORE_DOCS;
        }
        if (primed) {
          current = shared.nextDoc();
        } else {
          // The cursor already sits on the first document this output has not seen.
          primed = true;
        }
        if (current == NO_MORE_DOCS || current >= hi) {
          exhausted = true;
          return doc = NO_MORE_DOCS;
        }
        assert current >= lo : "cursor " + current + " is below output start " + lo;
        return doc = current - lo;
      }

      @Override
      public int advance(int target) throws IOException {
        int d;
        while ((d = nextDoc()) != NO_MORE_DOCS && d < target) {}
        return d;
      }

      @Override
      public long cost() {
        return shared.cost();
      }

      @Override
      public int freq() throws IOException {
        return shared.freq();
      }

      @Override
      public int nextPosition() throws IOException {
        return shared.nextPosition();
      }

      @Override
      public int startOffset() throws IOException {
        return shared.startOffset();
      }

      @Override
      public int endOffset() throws IOException {
        return shared.endOffset();
      }

      @Override
      public BytesRef getPayload() throws IOException {
        return shared.getPayload();
      }
    }
  }
}
