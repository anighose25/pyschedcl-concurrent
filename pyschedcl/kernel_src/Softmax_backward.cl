kernel void softmax_backward(
        const int mini_batch,
        const int class,
        global const float *predicted,
        global const float *actual,
        global float *output)
{
    const int global_id = get_global_id(0);

    if(global_id < mini_batch*class)
    {
       printf("%f - %f\n", predicted[global_id], actual[global_id]);
       output[global_id] = predicted[global_id] - actual[global_id];

    }
}
