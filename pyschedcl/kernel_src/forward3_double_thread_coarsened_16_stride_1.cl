

























void kernel forward_3_by_n_outplane(const int batchSize,
    global const double *images, global const double *filters, 
    global double *output,
    local double *_upstreamImage, local double *_filterCube) 
{
    const int globalId = get_global_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = 2*get_local_size(0);
    const int n = workgroupId / gNumFilters;
    const int outPlane = workgroupId % gNumFilters;

    const int localId0 = (get_local_id(0)/1)*1*2 +get_local_id(0)%1+0*1;
    int outputRow0 = localId0/gOutputSize;
    int outputCol0 = localId0%gOutputSize;
    const int minu0 = gPadZeros ? max(-gHalfFilterSize, -outputRow0) : -gHalfFilterSize;
    const int maxu0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow0  - gEven) : gHalfFilterSize - gEven;
    const int minv0 = gPadZeros ? max(-gHalfFilterSize, -outputCol0) : - gHalfFilterSize;
    const int maxv0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol0 - gEven) : gHalfFilterSize - gEven;

    const int localId1 = (get_local_id(0)/1)*1*2 +get_local_id(0)%1+1*1;
    int outputRow1 = localId1/gOutputSize;
    int outputCol1 = localId1%gOutputSize;
    const int minu1 = gPadZeros ? max(-gHalfFilterSize, -outputRow1) : -gHalfFilterSize;
    const int maxu1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow1  - gEven) : gHalfFilterSize - gEven;
    const int minv1 = gPadZeros ? max(-gHalfFilterSize, -outputCol1) : - gHalfFilterSize;
    const int maxv1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol1 - gEven) : gHalfFilterSize - gEven;

    const int numUpstreamsPerThread = (gInputSizeSquared + workgroupSize - 1) / workgroupSize;
    const int filterCubeLength = gInputPlanes * gFilterSizeSquared;
    const int filterCubeGlobalOffset = outPlane * filterCubeLength;
    const int numPixelsPerThread = (filterCubeLength + workgroupSize - 1) / workgroupSize;
    for (int i = 0; i < numPixelsPerThread; i++) 
    {

        int thisOffset0 = localId0 + i * workgroupSize;
		if(thisOffset0 < filterCubeLength)
			_filterCube[thisOffset0] = filters[filterCubeGlobalOffset + thisOffset0];


        int thisOffset1 = localId1 + i * workgroupSize;
		if(thisOffset1 < filterCubeLength)
			_filterCube[thisOffset1] = filters[filterCubeGlobalOffset + thisOffset1];


    }

    float sum0 = 0;

    float sum1 = 0;


    for (int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++) {
        int thisUpstreamImageOffset = (n * gInputPlanes + upstreamPlane) * gInputSizeSquared;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < numUpstreamsPerThread; i++) {
    
        int thisOffset0 = workgroupSize * i + localId0;
	    if (thisOffset0 < gInputSizeSquared)
		_upstreamImage[ thisOffset0 ] = images[ thisUpstreamImageOffset + thisOffset0 ];

    
        int thisOffset1 = workgroupSize * i + localId1;
	    if (thisOffset1 < gInputSizeSquared)
		_upstreamImage[ thisOffset1 ] = images[ thisUpstreamImageOffset + thisOffset1 ];

    
    }
        barrier(CLK_LOCAL_MEM_FENCE);
        int filterImageOffset = upstreamPlane * gFilterSizeSquared;
    
        for (int u = minu0; u <= maxu0; u++) {
            int inputRow = outputRow0 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv0; v <= maxv0; v++) {
                int inputCol = outputCol0 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId0 < gOutputSizeSquared) {
                   sum0 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu1; u <= maxu1; u++) {
            int inputRow = outputRow1 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv1; v <= maxv1; v++) {
                int inputCol = outputCol1 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId1 < gOutputSizeSquared) {
                   sum1 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
    }

    int resultIndex0 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId0;
    if (localId0 < gOutputSizeSquared)
	 output[resultIndex0] = sum0;
 

    int resultIndex1 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId1;
    if (localId1 < gOutputSizeSquared)
	 output[resultIndex1] = sum1;
 

}

  


