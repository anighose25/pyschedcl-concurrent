// from SpatialConvolutionMM.cu:

// CL: grid stride looping
#define CL_KERNEL_LOOP(i, n)                        \
  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \
      i < (n);                                       \
      i += get_local_size(0) * get_num_groups(0))

//#define gPadding gPadding
//#define gStride gStride
//#define gColSize gColSize
//#define gFilterSize gFilterSize
//#define gSize gSize

// Kernel for fast unfold+copy
// (adapted from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
kernel void im2col(
    const int n,
    global float const * im_data, int im_offset,
    global float* data_col)
{
  global const float *data_im = im_data + im_offset;

  CL_KERNEL_LOOP(index, n)
{
    int w_out = index % gColSize;
    index /= gColSize;
    int h_out = index % gColSize;
    int channel_in = index / gColSize;
    int channel_out = channel_in * gFilterSize  * gFilterSize ;
    int h_in = h_out * gStride - gPadding;
    int w_in = w_out * gStride - gPadding;
    data_col += (channel_out * gColSize + h_out) * gColSize + w_out;
    data_im += (channel_in * gSize + h_in) * gSize + w_in;
    for (int i = 0; i < gFilterSize ; ++i) {
      for (int j = 0; j < gFilterSize ; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < gSize && w < gSize) ?
          data_im[i * gSize + j] : 0;
        data_col += gColSize * gColSize;
      }
    }
  }
}

kernel void col2im(
    const int n,
    global float const *data_col,
    global float* im_data, int im_offset) {
  global float *data_im = im_data + im_offset;

  for (int index = get_group_id(0) * get_local_size(0) + get_local_id(0); index < (n); index += get_local_size(0) * get_num_groups(0)) {
    float val = 0;
    int w = index % gSize + gPadding;
    int h = (index / gSize) % gSize + gPadding;
    int c = index / (gSize * gSize);
    // compute the start and end of the output
    int w_col_start = (w < gFilterSize ) ? 0 : (w - gFilterSize ) / gStride + 1;
    int w_col_end = min(w / gStride + 1, gColSize);
    int h_col_start = (h < gFilterSize ) ? 0 : (h - gFilterSize ) / gStride + 1;
    int h_col_end = min(h / gStride + 1, gColSize);

    int offset = (c * gFilterSize  * gFilterSize  + h * gFilterSize  + w) * gColSize * gColSize;
    int coeff_h_col = (1 - gStride * gFilterSize  * gColSize) * gColSize;
    int coeff_w_col = (1 - gStride * gColSize * gColSize);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}
