__kernel void softmax(__global float *a, __global float *b, int num_cols)
{
  int i = get_global_id(0);
  float sum = 0;
  for(int j=0;j<num_cols;j++)
  {
    sum += exp(a[i*num_cols+j]);
  }
  for(int j=0;j<num_cols;j++)
  {
    b[i*num_cols+j] = exp(a[i*num_cols+j])/sum;
  }
}
