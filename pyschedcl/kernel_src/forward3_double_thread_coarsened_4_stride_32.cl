

























void kernel forward_3_by_n_outplane(const int batchSize,
    global const float *images, global const float *filters, 
    global float *output,
    local float *_upstreamImage, local float *_filterCube) 
{
    const int globalId = get_global_id(0);
    const int workgroupId = get_group_id(0);
    const int workgroupSize = 4*get_local_size(0);
    const int n = workgroupId / gNumFilters;
    const int outPlane = workgroupId % gNumFilters;

    const int localId0 = (get_local_id(0)/32)*32*4 +get_local_id(0)%32+0*32;
    int outputRow0 = localId0/gOutputSize;
    int outputCol0 = localId0%gOutputSize;
    const int minu0 = gPadZeros ? max(-gHalfFilterSize, -outputRow0) : -gHalfFilterSize;
    const int maxu0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow0  - gEven) : gHalfFilterSize - gEven;
    const int minv0 = gPadZeros ? max(-gHalfFilterSize, -outputCol0) : - gHalfFilterSize;
    const int maxv0 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol0 - gEven) : gHalfFilterSize - gEven;

    const int localId1 = (get_local_id(0)/32)*32*4 +get_local_id(0)%32+1*32;
    int outputRow1 = localId1/gOutputSize;
    int outputCol1 = localId1%gOutputSize;
    const int minu1 = gPadZeros ? max(-gHalfFilterSize, -outputRow1) : -gHalfFilterSize;
    const int maxu1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow1  - gEven) : gHalfFilterSize - gEven;
    const int minv1 = gPadZeros ? max(-gHalfFilterSize, -outputCol1) : - gHalfFilterSize;
    const int maxv1 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol1 - gEven) : gHalfFilterSize - gEven;

    const int localId2 = (get_local_id(0)/32)*32*4 +get_local_id(0)%32+2*32;
    int outputRow2 = localId2/gOutputSize;
    int outputCol2 = localId2%gOutputSize;
    const int minu2 = gPadZeros ? max(-gHalfFilterSize, -outputRow2) : -gHalfFilterSize;
    const int maxu2 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow2  - gEven) : gHalfFilterSize - gEven;
    const int minv2 = gPadZeros ? max(-gHalfFilterSize, -outputCol2) : - gHalfFilterSize;
    const int maxv2 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol2 - gEven) : gHalfFilterSize - gEven;

    const int localId3 = (get_local_id(0)/32)*32*4 +get_local_id(0)%32+3*32;
    int outputRow3 = localId3/gOutputSize;
    int outputCol3 = localId3%gOutputSize;
    const int minu3 = gPadZeros ? max(-gHalfFilterSize, -outputRow3) : -gHalfFilterSize;
    const int maxu3 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputRow3  - gEven) : gHalfFilterSize - gEven;
    const int minv3 = gPadZeros ? max(-gHalfFilterSize, -outputCol3) : - gHalfFilterSize;
    const int maxv3 = gPadZeros ? min(gHalfFilterSize - gEven, gOutputSize - 1 - outputCol3 - gEven) : gHalfFilterSize - gEven;

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


        int thisOffset2 = localId2 + i * workgroupSize;
		if(thisOffset2 < filterCubeLength)
			_filterCube[thisOffset2] = filters[filterCubeGlobalOffset + thisOffset2];


        int thisOffset3 = localId3 + i * workgroupSize;
		if(thisOffset3 < filterCubeLength)
			_filterCube[thisOffset3] = filters[filterCubeGlobalOffset + thisOffset3];


    }

    float sum0 = 0;

    float sum1 = 0;

    float sum2 = 0;

    float sum3 = 0;


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

    
        int thisOffset2 = workgroupSize * i + localId2;
	    if (thisOffset2 < gInputSizeSquared)
		_upstreamImage[ thisOffset2 ] = images[ thisUpstreamImageOffset + thisOffset2 ];

    
        int thisOffset3 = workgroupSize * i + localId3;
	    if (thisOffset3 < gInputSizeSquared)
		_upstreamImage[ thisOffset3 ] = images[ thisUpstreamImageOffset + thisOffset3 ];

    
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
    
        for (int u = minu2; u <= maxu2; u++) {
            int inputRow = outputRow2 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv2; v <= maxv2; v++) {
                int inputCol = outputCol2 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId2 < gOutputSizeSquared) {
                   sum2 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
                }
            } 
        }
    
        for (int u = minu3; u <= maxu3; u++) {
            int inputRow = outputRow3 + u;
            #if gPadZeros == 0
                inputRow += gHalfFilterSize;
            #endif
            int inputimagerowoffset = inputRow * gInputSize;
            int filterrowoffset = filterImageOffset + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
            for (int v = minv3; v <= maxv3; v++) {
                int inputCol = outputCol3 + v;
                #if gPadZeros == 0
                    inputCol += gHalfFilterSize;
                #endif
                if (localId3 < gOutputSizeSquared) {
                   sum3 += _upstreamImage[ inputimagerowoffset + inputCol] * _filterCube[ filterrowoffset + v ];
                   
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
 

    int resultIndex2 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId2;
    if (localId2 < gOutputSizeSquared)
	 output[resultIndex2] = sum2;
 

    int resultIndex3 = (n * gNumFilters + outPlane) * gOutputSizeSquared + localId3;
    if (localId3 < gOutputSizeSquared)
	 output[resultIndex3] = sum3;
 

}

  


