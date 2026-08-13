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

import org.apache.lucene.codecs.FilterCodec;
import org.apache.lucene.codecs.PointsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;

/**
 * The default codec with its points stored per range.
 *
 * <p>A segment records only its codec's name, so a reader resolves the points format through the
 * service loader rather than from whatever the writer was configured with. A test codec therefore
 * has to be a named, registered codec: wrapping the default one in place would be written down as
 * the default one and read back with a stock points reader.
 */
public final class PerRangePointsTestCodec extends FilterCodec {

  static final int NUM_RANGES = 8;

  private final PointsFormat pointsFormat;

  public PerRangePointsTestCodec() {
    super("PerRangePointsTest", new Lucene104Codec());
    this.pointsFormat =
        new PerRangePointsFormat(
            delegate.pointsFormat(), PerRangePointsFormat.equalSpans(NUM_RANGES));
  }

  @Override
  public PointsFormat pointsFormat() {
    return pointsFormat;
  }
}
