






void copyLocal(local float *target, global float const *source, int N) {

     
    int numLoops0 = (N + get_local_size(0) * 2 - 1) / get_local_size(0) * 2;
    for (int loop = 0; loop < numLoops0; loop++) {
        int offset0 = loop * get_local_size(0) * 2 + get_local_id(0) * 2;
        if (offset0 < N) {
            target[offset0] = source[offset0];
        }
    }


}

#ifdef gOutputSize
void kernel forward_4_by_n_outplane_smallercache(const int batchSize,
      global const float *images, global const float *filters,
    global float *output,
    local float *_inputPlane, local float *_filterPlane) {

     
    #define globalId0 get_group_id(0)*get_local_size(0)+(get_local_id(0)/32)*32*2 +get_local_id(0)%32+0*32 
    #define localId0 get_local_id(0)%32+0*32 
    #define workgroupId0 get_group_id(0) 
    
    const int effectiveWorkgroupId0 = workgroupId0 / gPixelsPerThread;
    const int pixel0 = workgroupId0 % gPixelsPerThread;
    const int effectiveLocalId0 = localId0 + pixel0 * gWorkgroupSize;
    const int n0 = effectiveWorkgroupId0 / gNumFilters;
    const int outPlane0 = effectiveWorkgroupId0 % gNumFilters;

    const int outputRow0 = effectiveLocalId0 / gOutputSize;
    const int outputCol0 = effectiveLocalId0 % gOutputSize;

    float sum0 = 0;
    for (int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        copyLocal(_inputPlane, images + (n0 * gInputPlanes + upstreamPlane) * gInputSizeSquared, gInputSizeSquared);
        copyLocal(_filterPlane, filters + (outPlane0 * gInputPlanes + upstreamPlane) * gFilterSizeSquared, gFilterSizeSquared);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (effectiveLocalId0 < gOutputSizeSquared) {
            for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {
                #if gPadZeros == 1
                    #define inputRow0 (outputRow0 + u)
                #else
                    #define inputRow0 (outputRow0 + u + gHalfFilterSize)
                #endif
                int inputimagerowoffset0 = inputRow0 * gInputSize;
                int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                bool rowOk0 = inputRow0 >= 0 && inputRow0 < gInputSize;
                for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {
                    #if gPadZeros == 1
                        #define inputCol0 (outputCol0 + v)
                    #else
                        #define inputCol0 (outputCol0 + v + gHalfFilterSize)
                    #endif
                    bool process0 = rowOk0 && inputCol0 >= 0 && inputCol0 < gInputSize;
                    if (process0) {
                            sum0 += _inputPlane[ inputimagerowoffset0 + inputCol0] * _filterPlane[ filterrowoffset + v ];
                    }
                }
            }
        }
    }
    #define resultIndex0 (( n0 * gNumFilters + outPlane0) * gOutputSizeSquared + effectiveLocalId0)
    if (effectiveLocalId0 < gOutputSizeSquared) {
        output[resultIndex0 ] = sum0;
    }


     
    #define globalId1 get_group_id(0)*get_local_size(0)+(get_local_id(0)/32)*32*2 +get_local_id(0)%32+1*32 
    #define localId1 get_local_id(0)%32+1*32 
    #define workgroupId1 get_group_id(0) 
    
    const int effectiveWorkgroupId1 = workgroupId1 / gPixelsPerThread;
    const int pixel1 = workgroupId1 % gPixelsPerThread;
    const int effectiveLocalId1 = localId1 + pixel1 * gWorkgroupSize;
    const int n1 = effectiveWorkgroupId1 / gNumFilters;
    const int outPlane1 = effectiveWorkgroupId1 % gNumFilters;

    const int outputRow1 = effectiveLocalId1 / gOutputSize;
    const int outputCol1 = effectiveLocalId1 % gOutputSize;

    float sum1 = 0;
    for (int upstreamPlane = 0; upstreamPlane < gInputPlanes; upstreamPlane++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        copyLocal(_inputPlane, images + (n1 * gInputPlanes + upstreamPlane) * gInputSizeSquared, gInputSizeSquared);
        copyLocal(_filterPlane, filters + (outPlane1 * gInputPlanes + upstreamPlane) * gFilterSizeSquared, gFilterSizeSquared);
        barrier(CLK_LOCAL_MEM_FENCE);

        if (effectiveLocalId1 < gOutputSizeSquared) {
            for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {
                #if gPadZeros == 1
                    #define inputRow1 (outputRow1 + u)
                #else
                    #define inputRow1 (outputRow1 + u + gHalfFilterSize)
                #endif
                int inputimagerowoffset1 = inputRow1 * gInputSize;
                int filterrowoffset = (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;
                bool rowOk1 = inputRow1 >= 0 && inputRow1 < gInputSize;
                for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {
                    #if gPadZeros == 1
                        #define inputCol1 (outputCol1 + v)
                    #else
                        #define inputCol1 (outputCol1 + v + gHalfFilterSize)
                    #endif
                    bool process1 = rowOk1 && inputCol1 >= 0 && inputCol1 < gInputSize;
                    if (process1) {
                            sum1 += _inputPlane[ inputimagerowoffset1 + inputCol1] * _filterPlane[ filterrowoffset + v ];
                    }
                }
            }
        }
    }
    #define resultIndex1 (( n1 * gNumFilters + outPlane1) * gOutputSizeSquared + effectiveLocalId1)
    if (effectiveLocalId1 < gOutputSizeSquared) {
        output[resultIndex1 ] = sum1;
    }



}
#endif
