#define TILE_DIM 32


__kernel void
naive_copy(__global float *in,
			 __global float* out,
       int n
       )
{
  int tx = get_global_id(0);
  int ty = get_global_id(1);


  out[tx*n+ty] = in[tx*n+ty];



}

__kernel void
naive_transpose(__global float *in,
			 __global float* out,
       int n
       )
{
  int tx = get_global_id(0);
  int ty = get_global_id(1);

  out[ty*n+tx] = in[tx*n+ty];



}

__kernel void
coalesced_transpose(__global float *in,
			 __global float* out,
       __local float * block, int m,
       int n
       )
{


  int tx = get_global_id(0);
  int ty = get_global_id(1);

	if (tx < n && ty < m){

		int lx = get_local_id(0);
  	int ly = get_local_id(1);

  	block[ly*(TILE_DIM+1)+lx] = in[ty*n+tx];


  	barrier(CLK_LOCAL_MEM_FENCE);

  	tx = get_group_id(0)*TILE_DIM + ly;
  	ty = get_group_id(1)*TILE_DIM + lx;

		if (tx < n && ty < m)
  		out[tx*m+ty] = block[lx*(TILE_DIM+1)+ly];

	}


}


__kernel void coalesced_transpose2(__global float *idata, __global float *odata, __local float * block, int n)
{
	// read the matrix tile into shared memory

	unsigned int xIndex = get_global_id(0);
	unsigned int yIndex = get_global_id(1);

	if((xIndex  < n) && (yIndex < n))
	{
		unsigned int index_in = yIndex * n + xIndex;
		block[get_local_id(1)*(TILE_DIM+1)+get_local_id(0)] = idata[index_in];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// write the transposed matrix tile to global memory
	xIndex = get_group_id(1) * TILE_DIM + get_local_id(0);
	yIndex = get_group_id(0) * TILE_DIM + get_local_id(1);
	if((xIndex < n) && (yIndex< n))
    {
		unsigned int index_out = yIndex * n + xIndex;
		odata[index_out] = block[get_local_id(0)*(TILE_DIM+1)+get_local_id(1)];
	}
}



__kernel void
coalesced_copy(__global float *in,
			 __global float* out,
       int n
       )
 {
   int tx = get_global_id(0);
   int ty = get_global_id(1);

   out[ty*n+tx] = in[ty*n+tx];

 }
