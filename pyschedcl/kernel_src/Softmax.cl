kernel void softmax_forward(
        const int mini_batch,
        const int class,
        global const float *data_input,
        global float *data_output)
{
    const int col = get_global_id(0);
    local float sum[BATCH_SIZE];
    local float maxx[BATCH_SIZE];
    sum[col] = 0;
    maxx[col] = data_input[col*class];
    
    

    for(int i=0;i<class;i++){
          maxx[col] = (maxx[col] > data_input[col*class + i]) ? maxx[col] : data_input[col*class + i];
    }
    
    

    for(int i=0;i<class;i++)
        data_output[col*class + i] = exp((data_input[col*class + i] - maxx[col]));
    
     
    
    for(int i=0;i<class;i++)
          sum[col]+=data_output[col*class + i];

    
    for(int i=0;i<class;i++)
        data_output[col*class + i] = data_output[col*class + i]/sum[col];

    for(int i=0;i<class;i++){
          printf("%f %f %f %f %d\n", data_input[col*class + i]  , maxx[col] , data_input[col*class + i],data_output[col*class + i] , col*class + i);
    }

}


kernel void softmax_backward(
        const int mini_batch,
        const int class,
        global const float *predicted,
        global const float *actual,
        global float *output)
{
    const int global_id = get_global_id(0);

    if(global_id < mini_batch*class)
      output[global_id] = predicted[global_id] - actual[global_id];

}
