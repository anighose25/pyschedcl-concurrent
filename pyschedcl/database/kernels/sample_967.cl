#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, const int c) {
  int d = get_global_id(0);

  if (d >= b - 1) {
    return;
  }
  b[d] += c;
}