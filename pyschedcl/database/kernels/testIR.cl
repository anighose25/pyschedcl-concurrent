 // OpenCL Kernel Function for testing LLVM IR
__kernel void testIR(__global const float* a, __global const float* b, __global float* c, int iNumElements)
{
    // get index into global data array
    int iGID = get_global_id(0);

    if (iGID >= iNumElements)
    {
        return;
    }

    // add the vector elements
    for(int i=0;i<iNumElements;i++)
	for(int j=0;j<iNumElements;j++)
	    c[iGID] = i*a[iGID] + j*b[iGID];
}
