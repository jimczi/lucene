package org.apache.lucene.sandbox.codecs.pq;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

import java.io.Closeable;
import java.io.IOException;

public class IndexInputFloatVectorValues extends FloatVectorValues implements RandomAccessVectorValues<float[]>, Closeable {
  private final int dims;
  private final int size;
  private final IndexInput input;
  private final float[] scratch;
  private int docID;

  public IndexInputFloatVectorValues(MMapDirectory directory, String fileName, int dims, int size) throws IOException {
    this.dims = dims;
    this.size = size;
    this.input = directory.openInput(fileName, IOContext.READ);
    this.scratch = new float[dims];
    this.docID = -1;
  }

  private IndexInputFloatVectorValues(IndexInput clone, int dims, int size) {
    this.dims = dims;
    this.size = size;
    this.input = clone;
    this.scratch = new float[dims];
    this.docID = -1;
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public float[] vectorValue() throws IOException {
    return vectorValue(docID);
  }

  @Override
  public int dimension() {
    return dims;
  }

  @Override
  public float[] vectorValue(int targetOrd) throws IOException {
    long pos = (long) targetOrd * dims * Float.BYTES;
    input.seek(pos);
    input.readFloats(scratch, 0, scratch.length);
    return scratch;
  }

  @Override
  public RandomAccessVectorValues<float[]> copy() throws IOException {
    return new IndexInputFloatVectorValues(input.clone(), dims, size);
  }

  @Override
  public int docID() {
    return docID;
  }

  @Override
  public int nextDoc() throws IOException {
    if (docID+1 == size) {
      return NO_MORE_DOCS;
    }
    return ++docID;
  }

  @Override
  public int advance(int target) throws IOException {
    this.docID = target < size ? target : NO_MORE_DOCS;
    return docID;
  }

  @Override
  public void close() throws IOException {
    IOUtils.close(input);
  }
}
