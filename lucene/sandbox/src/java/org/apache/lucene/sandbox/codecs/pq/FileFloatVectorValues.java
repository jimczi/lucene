package org.apache.lucene.sandbox.codecs.pq;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.hnsw.RandomAccessVectorValues;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;

public class FileFloatVectorValues extends FloatVectorValues implements RandomAccessVectorValues<float[]> {
  private final int dims;
  private final int size;
  private final FloatBuffer content;
  private final float[] scratch;

  private int docID;

  public FileFloatVectorValues(FileChannel input, int dims, int size) throws IOException {
    this.dims = dims;
    this.size = size;
    int bufferSize = dims * Float.BYTES;
    ByteBuffer buffer = input.map(FileChannel.MapMode.READ_ONLY, 0, (long) size * bufferSize)
            .order(ByteOrder.LITTLE_ENDIAN);
    this.content = buffer.asFloatBuffer();
    this.scratch = new float[dims];
    this.docID = -1;
  }

  private FileFloatVectorValues(FloatBuffer content, int dims, int size) {
    this.dims = dims;
    this.size = size;
    this.content = content;
    this.scratch = new float[dims];
    this.docID = -1;
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public float[] vectorValue() throws IOException {
    return new float[0];
  }

  @Override
  public int dimension() {
    return dims;
  }

  @Override
  public float[] vectorValue(int targetOrd) throws IOException {
    int pos = targetOrd * dims;
    content.get(pos, scratch);
    return scratch;
  }

  @Override
  public RandomAccessVectorValues<float[]> copy() throws IOException {
    return new FileFloatVectorValues(content, dims, size);
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
}
