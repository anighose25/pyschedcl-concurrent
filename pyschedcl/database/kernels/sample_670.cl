#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, __global float* c, const int d) {
  int e = get_global_id(0);

  if (e < c) {
    c[e] = a[e] + b[e];
  }
  b[e] = 0.0;
  c[e] = 0.0f;
  b[e] = c[e];
  c[e] = d;
}