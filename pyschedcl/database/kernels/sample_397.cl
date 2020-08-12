#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, __global float* c, __global float* d, const int e) {
  const int f = get_global_id(0);
  const int g = get_global_id(1);

  if (e < g && g < e) {
    c[f] = 0.0f;
    c[e] = 0.0f;
  }
  b[f] = c[f] * d[f];
}