





kernel void forwardNaive(const int batchSize, global const float *input, global int *selectors, global float *output) 
{

    const int globalId0 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/64)*64*4 +get_local_id(0)%64+0*64;

    const int globalId1 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/64)*64*4 +get_local_id(0)%64+1*64;

    const int globalId2 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/64)*64*4 +get_local_id(0)%64+2*64;

    const int globalId3 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/64)*64*4 +get_local_id(0)%64+3*64;

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

    const int intraImageOffset2 = globalId2 % gOutputSizeSquared;
    const int outputRow2 = intraImageOffset2 / gOutputSize;
    const int outputCol2 = intraImageOffset2 % gOutputSize;

    const int image2dIdx2 = globalId2 / gOutputSizeSquared;
    const int plane2 = image2dIdx2 % gNumPlanes;
    const int n2 = image2dIdx2 / gNumPlanes;

    if (n2 >= batchSize) {
        return;
    }

    inputRow = outputRow2 * gPoolingSize;
    inputCol = outputCol2 * gPoolingSize;
    inputImageOffset = (n2* gNumPlanes + plane2) * gInputSizeSquared;
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
    output[ globalId2 ] = maxValue;
    selectors[ globalId2 ] = selector;

    const int intraImageOffset3 = globalId3 % gOutputSizeSquared;
    const int outputRow3 = intraImageOffset3 / gOutputSize;
    const int outputCol3 = intraImageOffset3 % gOutputSize;

    const int image2dIdx3 = globalId3 / gOutputSizeSquared;
    const int plane3 = image2dIdx3 % gNumPlanes;
    const int n3 = image2dIdx3 / gNumPlanes;

    if (n3 >= batchSize) {
        return;
    }

    inputRow = outputRow3 * gPoolingSize;
    inputCol = outputCol3 * gPoolingSize;
    inputImageOffset = (n3* gNumPlanes + plane3) * gInputSizeSquared;
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
    output[ globalId3 ] = maxValue;
    selectors[ globalId3 ] = selector;
    
   
}
