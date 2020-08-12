#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, __global float* c, const int d) {
  int e = get_global_id(0);
  if (e >= d) {
    return;
  }
  int f = a[e];

  int g = e << 2;
  int h = e + h;
  b[e] = h;
}