





kernel void forwardNaive(const int batchSize, global const float *input, global int *selectors, global float *output) 
{

    const int globalId0 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/256)*256*2 +get_local_id(0)%256+0*256;

    const int globalId1 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/256)*256*2 +get_local_id(0)%256+1*256;

    int poolInputOffset,inputRow,inputCol,inputImageOffset,selector;
    float maxValue;

    const int intraImageOffset0 = globalId0 % gOutputSizeSquared;
    const int outputRow0 = intraImageOffset0 / gOutputSize;
    const int outputCol0 = intraImageOffset0 % gOutputSize;

    const int image2dIdx0 = globalId0 / gOutputSizeSquared;
    const int plane0 = image2dIdx0 % gNumPlanes;
    const int n0 = image2dIdx0 / gNumPlanes;

    if (n0 >= batchSize) {
        return;
    }

    inputRow = outputRow0 * gPoolingSize;
    inputCol = outputCol0 * gPoolingSize;
    inputImageOffset = (n0* gNumPlanes + plane0) * gInputSizeSquared;
    selector = 0;
    poolInputOffset = inputImageOffset + inputRow * gInputSize + inputCol;
    maxValue = input[ poolInputOffset ];
    for (int dRow = 0; dRow < gPoolingSize; dRow++) {
        for (int dCol = 0; dCol < gPoolingSize; dCol++) {
            bool process = (inputRow + dRow < gInputSize) && (inputCol + dCol < gInputSize);
            if (process) {
                float thisValue = input[ poolInputOffset + dRow * gInputSize + dCol ];
                if (thisValue > maxValue) {
                    maxValue = thisValue;
                    selector = dRow * gPoolingSize + dCol;
                }
            }
        }
    }
    output[ globalId0 ] = maxValue;
    selectors[ globalId0 ] = selector;

    const int intraImageOffset1 = globalId1 % gOutputSizeSquared;
    const int outputRow1 = intraImageOffset1 / gOutputSize;
    const int outputCol1 = intraImageOffset1 % gOutputSize;

    const int image2dIdx1 = globalId1 / gOutputSizeSquared;
    const int plane1 = image2dIdx1 % gNumPlanes;
    const int n1 = image2dIdx1 / gNumPlanes;

    if (n1 >= batchSize) {
        return;
    }

    inputRow = outputRow1 * gPoolingSize;
    inputCol = outputCol1 * gPoolingSize;
    inputImageOffset = (n1* gNumPlanes + plane1) * gInputSizeSquared;
    selector = 0;
    poolInputOffset = inputImageOffset + inputRow * gInputSize + inputCol;
    maxValue = input[ poolInputOffset ];
    for (int dRow = 0; dRow < gPoolingSize; dRow++) {
        for (int dCol = 0; dCol < gPoolingSize; dCol++) {
            bool process = (inputRow + dRow < gInputSize) && (inputCol + dCol < gInputSize);
            if (process) {
                float thisValue = input[ poolInputOffset + dRow * gInputSize + dCol ];
                if (thisValue > maxValue) {
                    maxValue = thisValue;
                    selector = dRow * gPoolingSize + dCol;
                }
            }
        }
    }
    output[ globalId1 ] = maxValue;
    selectors[ globalId1 ] = selector;
    
   
}
