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
import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.util.BytesRef;

/**
 * A doc values producer whose iterators cover only {@code [start, end)} of the segment.
 *
 * <p>Marking the other documents deleted is not enough to keep a merge from reading them. A merge
 * walks each field's values with {@link DocValuesIterator#nextDoc()} and drops whatever the
 * document map sends to {@code -1}, so the values are decoded first and discarded afterwards; an
 * output of a partitioned merge that wants a twentieth of the segment still pays for all of it, and
 * k outputs pay k times over. Seeking to the range and stopping at its end turns that back into one
 * read of the whole segment shared between the outputs.
 *
 * <p>Only the iteration is restricted. The value space -- the term dictionary behind a sorted field
 * and the ordinals into it -- is left whole, because a merge builds its ordinal map from it and
 * expects the same dictionary a full reader would have shown.
 */
final class DocRangeDocValuesProducer extends DocValuesProducer {

  private final DocValuesProducer in;
  private final int start;
  private final int end;

  DocRangeDocValuesProducer(DocValuesProducer in, int start, int end) {
    this.in = in;
    this.start = start;
    this.end = end;
  }

  /**
   * Positions {@code values} at its first document inside the range.
   *
   * @return that document, or {@link DocValuesIterator#NO_MORE_DOCS} if the range holds none
   */
  private int firstInRange(DocValuesIterator values) throws IOException {
    final int doc = values.advance(start);
    return doc >= end ? DocValuesIterator.NO_MORE_DOCS : doc;
  }

  private int nextInRange(DocValuesIterator values) throws IOException {
    final int doc = values.nextDoc();
    return doc >= end ? DocValuesIterator.NO_MORE_DOCS : doc;
  }

  private int advanceInRange(DocValuesIterator values, int target) throws IOException {
    final int doc = values.advance(Math.max(target, start));
    return doc >= end ? DocValuesIterator.NO_MORE_DOCS : doc;
  }

  private boolean inRange(int target) {
    return target >= start && target < end;
  }

  @Override
  public NumericDocValues getNumeric(FieldInfo field) throws IOException {
    final NumericDocValues values = in.getNumeric(field);
    if (values == null) {
      return null;
    }
    return new NumericDocValues() {
      private int doc = -1;

      @Override
      public int docID() {
        return doc;
      }

      @Override
      public int nextDoc() throws IOException {
        return doc = doc < start ? firstInRange(values) : nextInRange(values);
      }

      @Override
      public int advance(int target) throws IOException {
        return doc = advanceInRange(values, target);
      }

      @Override
      public boolean advanceExact(int target) throws IOException {
        if (inRange(target) == false) {
          return false;
        }
        doc = target;
        return values.advanceExact(target);
      }

      @Override
      public long cost() {
        return values.cost();
      }

      @Override
      public long longValue() throws IOException {
        return values.longValue();
      }
    };
  }

  @Override
  public BinaryDocValues getBinary(FieldInfo field) throws IOException {
    final BinaryDocValues values = in.getBinary(field);
    if (values == null) {
      return null;
    }
    return new BinaryDocValues() {
      private int doc = -1;

      @Override
      public int docID() {
        return doc;
      }

      @Override
      public int nextDoc() throws IOException {
        return doc = doc < start ? firstInRange(values) : nextInRange(values);
      }

      @Override
      public int advance(int target) throws IOException {
        return doc = advanceInRange(values, target);
      }

      @Override
      public boolean advanceExact(int target) throws IOException {
        if (inRange(target) == false) {
          return false;
        }
        doc = target;
        return values.advanceExact(target);
      }

      @Override
      public long cost() {
        return values.cost();
      }

      @Override
      public BytesRef binaryValue() throws IOException {
        return values.binaryValue();
      }
    };
  }

  @Override
  public SortedDocValues getSorted(FieldInfo field) throws IOException {
    final SortedDocValues values = in.getSorted(field);
    if (values == null) {
      return null;
    }
    return new SortedDocValues() {
      private int doc = -1;

      @Override
      public int docID() {
        return doc;
      }

      @Override
      public int nextDoc() throws IOException {
        return doc = doc < start ? firstInRange(values) : nextInRange(values);
      }

      @Override
      public int advance(int target) throws IOException {
        return doc = advanceInRange(values, target);
      }

      @Override
      public boolean advanceExact(int target) throws IOException {
        if (inRange(target) == false) {
          return false;
        }
        doc = target;
        return values.advanceExact(target);
      }

      @Override
      public long cost() {
        return values.cost();
      }

      @Override
      public int ordValue() throws IOException {
        return values.ordValue();
      }

      @Override
      public BytesRef lookupOrd(int ord) throws IOException {
        return values.lookupOrd(ord);
      }

      @Override
      public int getValueCount() {
        return values.getValueCount();
      }

      @Override
      public TermsEnum termsEnum() throws IOException {
        return values.termsEnum();
      }
    };
  }

  @Override
  public SortedNumericDocValues getSortedNumeric(FieldInfo field) throws IOException {
    final SortedNumericDocValues values = in.getSortedNumeric(field);
    if (values == null) {
      return null;
    }
    return new SortedNumericDocValues() {
      private int doc = -1;

      @Override
      public int docID() {
        return doc;
      }

      @Override
      public int nextDoc() throws IOException {
        return doc = doc < start ? firstInRange(values) : nextInRange(values);
      }

      @Override
      public int advance(int target) throws IOException {
        return doc = advanceInRange(values, target);
      }

      @Override
      public boolean advanceExact(int target) throws IOException {
        if (inRange(target) == false) {
          return false;
        }
        doc = target;
        return values.advanceExact(target);
      }

      @Override
      public long cost() {
        return values.cost();
      }

      @Override
      public long nextValue() throws IOException {
        return values.nextValue();
      }

      @Override
      public int docValueCount() {
        return values.docValueCount();
      }
    };
  }

  @Override
  public SortedSetDocValues getSortedSet(FieldInfo field) throws IOException {
    final SortedSetDocValues values = in.getSortedSet(field);
    if (values == null) {
      return null;
    }
    return new SortedSetDocValues() {
      private int doc = -1;

      @Override
      public int docID() {
        return doc;
      }

      @Override
      public int nextDoc() throws IOException {
        return doc = doc < start ? firstInRange(values) : nextInRange(values);
      }

      @Override
      public int advance(int target) throws IOException {
        return doc = advanceInRange(values, target);
      }

      @Override
      public boolean advanceExact(int target) throws IOException {
        if (inRange(target) == false) {
          return false;
        }
        doc = target;
        return values.advanceExact(target);
      }

      @Override
      public long cost() {
        return values.cost();
      }

      @Override
      public long nextOrd() throws IOException {
        return values.nextOrd();
      }

      @Override
      public int docValueCount() {
        return values.docValueCount();
      }

      @Override
      public BytesRef lookupOrd(long ord) throws IOException {
        return values.lookupOrd(ord);
      }

      @Override
      public long getValueCount() {
        return values.getValueCount();
      }

      @Override
      public TermsEnum termsEnum() throws IOException {
        return values.termsEnum();
      }
    };
  }

  @Override
  public DocValuesSkipper getSkipper(FieldInfo field) {
    // Left alone: a skipper only ever narrows what a caller has to look at, so restricting it would
    // save nothing that the iterators above have not already saved.
    return in.getSkipper(field);
  }

  @Override
  public void checkIntegrity(MergePolicy.OneMerge merge) throws IOException {
    in.checkIntegrity(merge);
  }

  @Override
  public void close() throws IOException {
    in.close();
  }
}
