#define BLOCK_SIZE 32

__kernel void
lud_diagonal(__global float *m,
			 __local  float *shadow,
			 int   matrix_dim
       )
{
	int i,j;
	int tx = get_local_id(0);

	int array_offset = 0;
	for(i=0; i < BLOCK_SIZE; i++){
		shadow[i * BLOCK_SIZE + tx]=m[array_offset+tx];
		array_offset += matrix_dim;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for(i=0; i < BLOCK_SIZE-1; i++) {

    if (tx>i){
      for(j=0; j < i; j++)
        shadow[tx * BLOCK_SIZE + i] -= shadow[tx * BLOCK_SIZE + j] * shadow[j * BLOCK_SIZE + i];
		shadow[tx * BLOCK_SIZE + i] /= shadow[i * BLOCK_SIZE + i];
    }

	barrier(CLK_LOCAL_MEM_FENCE);
    if (tx>i){

      for(j=0; j < i+1; j++)
        shadow[(i+1) * BLOCK_SIZE + tx] -= shadow[(i+1) * BLOCK_SIZE + j]*shadow[j * BLOCK_SIZE + tx];
    }

	barrier(CLK_LOCAL_MEM_FENCE);
    }

    array_offset = matrix_dim;
    for(i=1; i < BLOCK_SIZE; i++){
      m[array_offset+tx]=shadow[i * BLOCK_SIZE + tx];
      array_offset += matrix_dim;
    }

}

__kernel void
lud_perimeter_row(__global float *m,
        __global float * diagonal_output,
			  __local  float *dia,
			  __local  float *peri_row,
			  int matrix_dim
        )
{
    //write a check to ensure that the thread isnt outside the matrix bounds
    int i,j, array_offset;
    int idx;

    int  bx = get_group_id(0);
    int  tx = get_local_id(0);

    idx = tx;

    array_offset = 0;
    for (i=0; i < BLOCK_SIZE; i++)
    {
      dia[i * BLOCK_SIZE + idx]=diagonal_output[array_offset+idx];
      array_offset += matrix_dim;
    }

    array_offset = 0;
    for (i=0; i < BLOCK_SIZE; i++)
    {
      peri_row[i * BLOCK_SIZE+ idx]=m[array_offset+bx*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }



    barrier(CLK_LOCAL_MEM_FENCE);


     idx=tx;
    for(i=1; i < BLOCK_SIZE; i++)
    {
      for (j=0; j < i; j++)
        peri_row[i * BLOCK_SIZE + idx]-=dia[i * BLOCK_SIZE+ j]*peri_row[j * BLOCK_SIZE + idx];
    }


	barrier(CLK_LOCAL_MEM_FENCE);


    idx=tx;
    array_offset = matrix_dim;
    for(i=1; i < BLOCK_SIZE; i++)
    {
      m[array_offset+bx*BLOCK_SIZE+idx] = peri_row[i*BLOCK_SIZE+idx];
      array_offset += matrix_dim;
    }


}

__kernel void
lud_perimeter_col(__global float *m,
        __global float * diagonal_output,
			  __local  float *dia,
			  __local  float *peri_col,
			  int matrix_dim
        )
{
    int i,j, array_offset;
    int idx;

    int  bx = get_group_id(0);
    int  tx = get_local_id(0);


    idx = tx;

    array_offset = 0;
    for (i=0; i < BLOCK_SIZE; i++){
      dia[i * BLOCK_SIZE + idx]=diagonal_output[array_offset+idx];
      array_offset += matrix_dim;
    }

    array_offset = bx*BLOCK_SIZE*BLOCK_SIZE;
    for (i=0; i < BLOCK_SIZE; i++) {
      peri_col[i * BLOCK_SIZE + idx] = m[array_offset+idx];
      array_offset += BLOCK_SIZE;
    }


    barrier(CLK_LOCAL_MEM_FENCE);



     idx=tx;
     for(i=0; i < BLOCK_SIZE; i++)
     {
        for(j=0; j < i; j++)
          peri_col[idx * BLOCK_SIZE + i]-=peri_col[idx * BLOCK_SIZE+ j]*dia[j * BLOCK_SIZE + i];

      peri_col[idx * BLOCK_SIZE + i] /= dia[i * BLOCK_SIZE+ i];

      }


	  barrier(CLK_LOCAL_MEM_FENCE);


    idx=tx;
    array_offset = bx*BLOCK_SIZE*BLOCK_SIZE;

    for(i=0; i < BLOCK_SIZE; i++)
    {
      m[array_offset+idx] =  peri_col[i*BLOCK_SIZE+idx];
      array_offset += BLOCK_SIZE;
    }


}

__kernel void
lud_internal(__global float *m,
			__global float* peri_row_output,
			__global float* peri_col_output,
			 __local  float *peri_row,
			 __local  float *peri_col,
			int matrix_dim_x)
{

  int  bx = get_group_id(0);
  int  by = get_group_id(1);

  int  tx = get_local_id(0);
  int  ty = get_local_id(1);

  int i;
  float sum;

  int global_row_id = by*BLOCK_SIZE;
  int global_col_id = bx*BLOCK_SIZE;

  peri_row[ty * BLOCK_SIZE + tx] = peri_row_output[(ty)*matrix_dim_x+global_col_id+tx];
  peri_col[ty * BLOCK_SIZE + tx] = peri_col_output[(global_row_id+ty)*BLOCK_SIZE+tx];

  barrier(CLK_LOCAL_MEM_FENCE);

  sum = 0;
  for (i=0; i < BLOCK_SIZE; i++)
    sum += peri_col[ty * BLOCK_SIZE + i] * peri_row[i * BLOCK_SIZE + tx];
  m[(global_row_id+ty)*matrix_dim_x+global_col_id+tx] -= sum;


}
