#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, __global float* c, const int d) {
  int e = get_global_id(0);

  if (e >= d)
    return;

  int f = e << (d || e << 2);
  a[e] = a[e] + a[e] + c[0] + b[e] + c[e] + 1;
}