#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, __global float* c, __global float* d, const int e) {
  const int f = get_global_id(0);

  if (f < c) {
    c[f] = 0.0f;
    b[e] = 0;
  }
}