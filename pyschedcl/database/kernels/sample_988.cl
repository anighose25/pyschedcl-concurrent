#include "opencl-shim.h"
__kernel void A(__global float* a, __global float* b, __global float* c, const int d) {
  int e = get_global_id(0);

  if (e < d) {
    if (e == 0) {
      c[e] = 3;
    }
  } else if (e < d) {
    c[e] = 0;
  }
}