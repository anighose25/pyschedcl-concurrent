__kernel void transpose(const int P, const int Q,
                        const __global float* input,
                        __global float* output) {

    // Thread identifiers
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int ID0 = get_group_id(0)*TRANSPOSEX + tx; // 0..P
    const int ID1 = get_group_id(1)*TRANSPOSEY + ty; // 0..Q
    //printf("%d %d\n",ID0,ID1);
    // Set-up the local memory for shuffling
    __local float buffer[TRANSPOSEX][TRANSPOSEY];

    // Swap the x and y coordinates to perform the rotation (coalesced)
    if (ID0 < P && ID1 < Q) {
        buffer[ty][tx] = input[ID1*P + ID0];
    }

    // Synchronise all threads
    barrier(CLK_LOCAL_MEM_FENCE);

    // We don't have to swap the x and y thread indices here,
    // because that's already done in the local memory
    const int newID0 = get_group_id(1)*TRANSPOSEY + tx;
    const int newID1 = get_group_id(0)*TRANSPOSEX + ty;

    // Store the transposed result (coalesced)
    if (newID0 < Q && newID1 < P) {
        output[newID1*Q + newID0] = buffer[tx][ty];
    }
}

__kernel void transpose_naive(int height, int width, __global float *idata, __global float* odata)
{
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);
   
    if (xIndex + offset < width && yIndex < height)
    {
        unsigned int index_in  = xIndex + offset + width * yIndex;
        unsigned int index_out = yIndex + height * xIndex;
        odata[index_out] = idata[index_in];
    }
}


__kernel void shared_transpose(int height, int width, __global float *idata, __global float *odata)
{
        // read the matrix tile into shared memory
        unsigned int xIndex = get_global_id(0);
        unsigned int yIndex = get_global_id(1);
	__local float buffer[BLOCK_DIM][BLOCK_DIM];

    	unsigned int index_in = yIndex * width + xIndex;
        if((xIndex < width) && (yIndex < height))
        {
                block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)] = idata[index_in];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if((xIndex < height) && (yIndex < width))
    	{
                odata[index_in] = block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)];
        }
}


