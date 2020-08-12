// concept: each workgroup handles convolving one input example with one filtercube
// and writing out one single output plane
//
// workgroup id organized like: [imageid][outplane]
// local id organized like: [outrow][outcol]
// each thread iterates over: [upstreamplane][filterrow][filtercol]
// number workgroups = 32
// one filter plane takes up 5 * 5 * 4 = 100 bytes
// one filter cube (corresponding to one outplane) = 5*5 * 32 * 4 = 3.2KB (ok)
// all filter cubes = 3.2KB * 32 = 102KB (too big)
// output are organized like [imageid][filterid][row][col]

void kernel forward_3_by_n_outplane(const int batchSize,
    global const float *images, global const float *filters,
    global float *output,
    local float *_upstreamImage, local float *_filterCube) {
    
    const int globalId = get_global_id(0);

    const int workgroupId = get_group_id(0);
    
    const int workgroupSize = get_local_size(0);
    
    const int n = workgroupId / gNumFilters;
    const int outPlane = workgroupId % gNumFilters;

    const int localId = get_local_id(0);
    
    const int outputRow = localId / gOutputSize;
    const int outputCol = localId % gOutputSize;
    
    // if(DEBUG) printf("%d %d writing output image %f\n",localId,workgroupId,images[localId]);

    const int minu = gPadZeros ? max(-gHalfFilterSize, -outputRow) : -gHalfFilterSize;
    const int maxu = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow  - gEven) : gHalfFilterSize - gEven;
    const int minv = gPadZeros ? max(-gHalfFilterSize, -outputCol) : - gHalfFilterSize;
    const int maxv = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol - gEven) : gHalfFilterSize - gEven;

    const int numUpstreamsPerThread = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;
    const int filterCubeLength = gInputPlanes * gFilterSizeSquared;
    const int filterCubeGlobalOffset = outPlane * filterCubeLength;
    const int numPixelsPerThread = (filterCubeLength + workgroupSize - 1) / workgroupSize;
    for (int i = 0; i < numPixelsPerThread; i++) {
        int thisOffset = localId + i * workgroupSize;
        if(DEBUG) printf("Accessing filtercube at offset %d\n", filterCubeGlobalOffset+thisOffset);
        if (thisOffset < filterCubeLength) {
            // _filterCube[thisOffset] = filters[(filterCubeGlobalOffset + thisOffset)];
        }
    }
    // if(DEBUG) printf("Thread %d of WG %d  for image %d at (%d,%d) loaded filter cube\n", localId, workgroupId,n,outputRow,outputCol);
    // dont need a barrier, since we'll just run behind the barrier from the upstream image download

    float sum = 0;
    for (int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++) {
        int thisUpstreamImageOffset = (n * gInputPlanes + upstreamPlane) * gInputSizeSquared;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < numUpstreamsPerThread; i++) {
            int thisOffset = workgroupSize * i + localId;
            if(DEBUG) printf("Accessing image at offset %d\n", thisUpstreamImageOffset+thisOffset);
            if (thisOffset < gInputSizeSquared) {
                // printf(" %d \n", (thisUpstreamImageOffset + thisOffset));
                  _upstreamImage[ thisOffset ] = images[ (thisUpstreamImageOffset + thisOffset)];
            }
        //printf("Image point %f\n",images[ thisUpstreamImageOffset + thisOffset ]);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        // if (DEBUG) printf("Thread %d of WG %d  for image %d at (%d,%d) loaded upstream image for plane %d \n", localId, workgroupId,n,outputRow,outputCol,upstreamPlane);
        int filterImageOffset = upstreamPlane * gFilterSizeSquared;
        for (int u = minu; u <= maxu; u++) {
            int inputRow = outputRow + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv; v <= maxv; v++) {
                int inputCol = outputCol + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId < gOutputSizeSquared) {
                //    if(DEBUG) printf("Thread %d of WG %d  for image %d at (%d,%d) starting computation at (%d,%d)-->(%d) of inputPlane %d with neighborhood (%d,%d,%d,%d) input element %f\n", localId, workgroupId,n,outputRow,outputCol,inputRow,inputCol,inputimagerowoffset + inputCol,upstreamPlane,minu,maxu,minv,maxv, _upstreamImage[ inputimagerowoffset + inputCol]);
                    sum += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   // printf("Thread %d of WG %d  for image %d at (%d,%d) finishing computation \n", localId, workgroupId,n,outputRow,outputCol);
                }
            }
        }
    }

    // output are organized like [imageid][filterid][row][col]
    //printf("Thread %d of WG %d  for image %d at (%d,%d) computing result index \n", localId, workgroupId,n,outputRow,outputCol);
    int resultIndex = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId;
    if (DEBUG) printf("thread %d Result Index %d\n",globalId,resultIndex);
    if (localId < gOutputSizeSquared) {
        if(DEBUG)printf("%d %d writing output %f\n",localId,workgroupId,sum);
        output[resultIndex ] = sum;
    }

}
