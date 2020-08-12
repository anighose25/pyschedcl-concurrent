





kernel void backward_pooling(const int batchSize, global const float *gradOutput, global const int *selectors, global float *gradInput) 
{

    const int globalId0 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/32)*32*2 +get_local_id(0)%32+0*32;

    const int globalId1 = get_group_id(0)*get_local_size(0)+(get_local_id(0)/32)*32*2 +get_local_id(0)%32+1*32;


     #define nPlaneCombo0 (globalId0 / gOutputSizeSquared) 
    #define outputPosCombo0 (globalId0 % gOutputSizeSquared)

    int n0 = nPlaneCombo0 / gNumPlanes;
    int plane0 = nPlaneCombo0 % gNumPlanes;
    int outputRow0 = outputPosCombo0 / gOutputSize;
    int outputCol0 = outputPosCombo0 % gOutputSize;

    if (n0 >= batchSize) {
        return;
    }

    int resultIndex0 = (( n0
        * gNumPlanes + plane0)
        * gOutputSize + outputRow0)
        * gOutputSize + outputCol0;
    #define error0 (gradOutput[resultIndex0])
    int selector0 = (selectors[resultIndex0]);
    #define drow0 (selector0 / gPoolingSize)
    #define dcol0 (selector0 % gPoolingSize)
    #define inputRow0 (outputRow0 * gPoolingSize + drow0)
    #define inputCol0 (outputCol0 * gPoolingSize + dcol0)
    int inputIndex0 = (( n0
        * gNumPlanes + plane0)
        * gInputSize + inputRow0)
        * gInputSize + inputCol0;
    gradInput[ inputIndex0 ] = error0;

     #define nPlaneCombo1 (globalId1 / gOutputSizeSquared) 
    #define outputPosCombo1 (globalId1 % gOutputSizeSquared)

    int n1 = nPlaneCombo1 / gNumPlanes;
    int plane1 = nPlaneCombo1 % gNumPlanes;
    int outputRow1 = outputPosCombo1 / gOutputSize;
    int outputCol1 = outputPosCombo1 % gOutputSize;

    if (n1 >= batchSize) {
        return;
    }

    int resultIndex1 = (( n1
        * gNumPlanes + plane1)
        * gOutputSize + outputRow1)
        * gOutputSize + outputCol1;
    #define error1 (gradOutput[resultIndex1])
    int selector1 = (selectors[resultIndex1]);
    #define drow1 (selector1 / gPoolingSize)
    #define dcol1 (selector1 % gPoolingSize)
    #define inputRow1 (outputRow1 * gPoolingSize + drow1)
    #define inputCol1 (outputCol1 * gPoolingSize + dcol1)
    int inputIndex1 = (( n1
        * gNumPlanes + plane1)
        * gInputSize + inputRow1)
        * gInputSize + inputCol1;
    gradInput[ inputIndex1 ] = error1;
    
 
    
}

