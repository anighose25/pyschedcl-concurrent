kernel void backward_pooling(const int batchSize,
    global const float *gradOutput, global const int *selectors, global float *gradInput) {

    #define globalId get_global_id(0)
    #define nPlaneCombo (globalId / gOutputSizeSquared)
    #define outputPosCombo (globalId % gOutputSizeSquared)

    const int n = nPlaneCombo / gNumPlanes;
    const int plane = nPlaneCombo % gNumPlanes;
    const int outputRow = outputPosCombo / gOutputSize;
    const int outputCol = outputPosCombo % gOutputSize;

    if (n >= batchSize) {
        return;
    }

    int resultIndex = (( n
        * gNumPlanes + plane)
        * gOutputSize + outputRow)
        * gOutputSize + outputCol;
    #define error (gradOutput[resultIndex])
    int selector = (selectors[resultIndex]);
    #define drow (selector / gPoolingSize)
    #define dcol (selector % gPoolingSize)
    #define inputRow (outputRow * gPoolingSize + drow)
    #define inputCol (outputCol * gPoolingSize + dcol)
    int inputIndex = (( n
        * gNumPlanes + plane)
        * gInputSize + inputRow)
        * gInputSize + inputCol;

    gradInput[ inputIndex ] = error;
}
