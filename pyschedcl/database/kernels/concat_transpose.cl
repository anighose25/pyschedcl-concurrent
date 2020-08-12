__kernel void
concat_transpose(__global float *m1,
			 __global  float *m2,
       __global float * out,
			 int  num_rows_m1,
       int  num_rows_m2,
       int num_cols

       )
{
  int j = get_global_id(0);
  int i = get_global_id(1);

  if (i < num_rows_m1)
  {
    out[j*(num_rows_m1+num_rows_m2)+i] = m1[i*num_cols+j];
    //printf("in first block %d %d %d %d %d\n",i,j,num_rows_m1,num_rows_m2,num_cols);
  }
  else
  {
    out[j*(num_rows_m1+num_rows_m2)+i] = m2[(i-num_rows_m1)*num_cols+j];
    //printf("in second block %d %d %d %d\n",i,j,num_rows_m1,num_rows_m2,num_cols);
  }
  //printf("%d %d",j,i);
}

__kernel void
concat(__global float *m1,
			 __global  float *m2,
       __global float * out,
			 int  num_rows_m1,
       int  num_rows_m2,
       int num_cols

       )
{
  int i = get_global_id(0);
  int j = get_global_id(1);

  if (i < num_rows_m1)
  {
    out[i*(num_cols)+j] = m1[i*num_cols+j];
    //printf("in first block %d %d %d %d %d\n",i,j,num_rows_m1,num_rows_m2,num_cols);
  }
  else
  {
    out[i*(num_cols)+j] = m2[(i-num_rows_m1)*num_cols+j];
    //printf("in second block %d %d %d %d\n",i,j,num_rows_m1,num_rows_m2,num_cols);
  }
  //printf("%d %d",j,i);
}
