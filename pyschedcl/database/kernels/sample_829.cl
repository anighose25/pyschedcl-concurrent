#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, __global float* c, const int d) {
  int e = get_global_id(0);
  if (e >= d) {
    return;
  }
  if (d < c) {
    a[e] += a[e];
  }
  b[e] = a[e] + a[e];
}