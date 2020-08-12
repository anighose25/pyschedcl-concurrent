





kernel void forwardNaive(const int batchSize, global const float *input, global int *selectors, global float *output) 
{

    const int globalId0 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/64)*64*8 +get_local_id(0)%64+0*64;

    const int globalId1 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/64)*64*8 +get_local_id(0)%64+1*64;

    const int globalId2 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/64)*64*8 +get_local_id(0)%64+2*64;

    const int globalId3 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/64)*64*8 +get_local_id(0)%64+3*64;

    const int globalId4 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/64)*64*8 +get_local_id(0)%64+4*64;

    const int globalId5 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/64)*64*8 +get_local_id(0)%64+5*64;

    const int globalId6 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/64)*64*8 +get_local_id(0)%64+6*64;

    const int globalId7 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/64)*64*8 +get_local_id(0)%64+7*64;

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

    const int intraImageOffset4 = globalId4 % gOutputSizeSquared;
    const int outputRow4 = intraImageOffset4 / gOutputSize;
    const int outputCol4 = intraImageOffset4 % gOutputSize;

    const int image2dIdx4 = globalId4 / gOutputSizeSquared;
    const int plane4 = image2dIdx4 % gNumPlanes;
    const int n4 = image2dIdx4 / gNumPlanes;

    if (n4 >= batchSize) {
        return;
    }

    inputRow = outputRow4 * gPoolingSize;
    inputCol = outputCol4 * gPoolingSize;
    inputImageOffset = (n4* gNumPlanes + plane4) * gInputSizeSquared;
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
    output[ globalId4 ] = maxValue;
    selectors[ globalId4 ] = selector;

    const int intraImageOffset5 = globalId5 % gOutputSizeSquared;
    const int outputRow5 = intraImageOffset5 / gOutputSize;
    const int outputCol5 = intraImageOffset5 % gOutputSize;

    const int image2dIdx5 = globalId5 / gOutputSizeSquared;
    const int plane5 = image2dIdx5 % gNumPlanes;
    const int n5 = image2dIdx5 / gNumPlanes;

    if (n5 >= batchSize) {
        return;
    }

    inputRow = outputRow5 * gPoolingSize;
    inputCol = outputCol5 * gPoolingSize;
    inputImageOffset = (n5* gNumPlanes + plane5) * gInputSizeSquared;
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
    output[ globalId5 ] = maxValue;
    selectors[ globalId5 ] = selector;

    const int intraImageOffset6 = globalId6 % gOutputSizeSquared;
    const int outputRow6 = intraImageOffset6 / gOutputSize;
    const int outputCol6 = intraImageOffset6 % gOutputSize;

    const int image2dIdx6 = globalId6 / gOutputSizeSquared;
    const int plane6 = image2dIdx6 % gNumPlanes;
    const int n6 = image2dIdx6 / gNumPlanes;

    if (n6 >= batchSize) {
        return;
    }

    inputRow = outputRow6 * gPoolingSize;
    inputCol = outputCol6 * gPoolingSize;
    inputImageOffset = (n6* gNumPlanes + plane6) * gInputSizeSquared;
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
    output[ globalId6 ] = maxValue;
    selectors[ globalId6 ] = selector;

    const int intraImageOffset7 = globalId7 % gOutputSizeSquared;
    const int outputRow7 = intraImageOffset7 / gOutputSize;
    const int outputCol7 = intraImageOffset7 % gOutputSize;

    const int image2dIdx7 = globalId7 / gOutputSizeSquared;
    const int plane7 = image2dIdx7 % gNumPlanes;
    const int n7 = image2dIdx7 / gNumPlanes;

    if (n7 >= batchSize) {
        return;
    }

    inputRow = outputRow7 * gPoolingSize;
    inputCol = outputCol7 * gPoolingSize;
    inputImageOffset = (n7* gNumPlanes + plane7) * gInputSizeSquared;
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
    output[ globalId7 ] = maxValue;
    selectors[ globalId7 ] = selector;
    
   
}
