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

/** Various third party contributions and new ideas */
module org.apache.lucene.sandbox {
  requires org.apache.lucene.core;
  requires org.apache.lucene.queries;
  requires jdk.incubator.vector;
  requires org.apache.lucene.misc;

  exports org.apache.lucene.payloads;
  exports org.apache.lucene.sandbox.codecs.idversion;
  exports org.apache.lucene.sandbox.document;
  exports org.apache.lucene.sandbox.queries;
  exports org.apache.lucene.sandbox.search;
  exports org.apache.lucene.sandbox.index;

  provides org.apache.lucene.codecs.PostingsFormat with
          org.apache.lucene.sandbox.codecs.idversion.IDVersionPostingsFormat;
}
