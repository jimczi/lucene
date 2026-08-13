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
package org.apache.lucene.codecs;

import java.io.Closeable;
import java.io.IOException;
import org.apache.lucene.index.TermsEnum;
import org.apache.lucene.util.BytesRef;

/**
 * Writes one field's terms in <i>push</i> order: the caller drives, handing over one term at a time.
 *
 * <p>{@link FieldsConsumer#write(org.apache.lucene.index.Fields, NormsProducer)} is pull-shaped --
 * the consumer walks the source itself -- which means <i>n</i> consumers fed from one source must
 * walk it <i>n</i> times. That is the dominant cost of a partitioned merge over an inverted index:
 * unlike doc values or stored fields, whose per-output doc range is a contiguous interval a reader
 * can seek past, a terms dictionary is ordered by term, so every output must enumerate the whole
 * dictionary and read every posting. Inverting the direction lets a single walk feed every output.
 *
 * <p>Terms must be pushed in increasing order within a field, exactly as {@code write} would have
 * produced them. Closing finishes the field; it does not close the owning {@link FieldsConsumer}.
 *
 * @lucene.experimental
 */
public interface TermsPushWriter extends Closeable {

  /**
   * Writes one term's postings. {@code source} must be positioned at {@code term}; the writer pulls
   * the postings from it, so a caller feeding several outputs from one source passes a view
   * restricted to the documents that output owns.
   */
  void write(BytesRef term, TermsEnum source, NormsProducer norms) throws IOException;

  /** Finishes this field. */
  @Override
  void close() throws IOException;
}
