/**
 * gemm.cl: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

typedef float DATA_TYPE;



__kernel void gemm(__global DATA_TYPE *a, __global DATA_TYPE *b, __global DATA_TYPE *c, DATA_TYPE alpha, DATA_TYPE beta, int ni, int nj, int nk)
{
  int j = get_global_id(1);
	int i = get_global_id(0);




	if ((i < ni) && (j < nj))
	{
		c[i * nj + j] *= beta;

    int k;
		for(k=0; k < nk; k++)
		{
      //printf("printing a[%d][%d] - %f and b[%d][%d] - %f\n",i,k,a[i*nk+k],k,j,b[k*nj+j]);
			c[i * nj + j] += alpha * a[i * nk + k] * b[k * nj +j];
		}
	}

  //printf("work item %d %d, result %f \n",i,j,c[i*nj+j]);


}

__kernel void coalesced_gemm(__global DATA_TYPE *a, __global DATA_TYPE *b, __global DATA_TYPE *c, DATA_TYPE alpha, DATA_TYPE beta, int ni, int nj, int nk)
{
  int ty = get_global_id(1);
	int tx = get_global_id(0);




	if ((tx < nj) && (ty < ni))
	{
		c[ty * nj + tx] *= beta;

    int k;
		for(k=0; k < nk; k++)
		{
      //printf("printing a[%d][%d] - %f and b[%d][%d] - %f\n",i,k,a[i*nk+k],k,j,b[k*nj+j]);
			c[ty * nj + tx] += alpha * a[ty * nk + k] * b[k * nj +tx];
		}
	}

  //printf("work item %d %d, result %f \n",i,j,c[i*nj+j]);


}
