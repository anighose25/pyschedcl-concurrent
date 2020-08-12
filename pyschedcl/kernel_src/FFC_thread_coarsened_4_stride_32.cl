










__kernel void myGEMM3(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

        
    int RTS=TS/4;

    const int row = get_local_id(0); // Local row ID (max: TS/W_PT == R_TS)
    const int col = get_local_id(1); // Local col ID (max: TS)
    const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
    const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

    __local float Asub[TS][TS];
    __local float Bsub[TS][TS];
    // Initialise the accumulation registers
    float acc[4];
    for (int w=0; w<4; w++) {
        acc[w] = 0.0f;
    }

    const int numTiles = K/TS;
    for (int t=0; t<numTiles; t++) {

        for (int w=0; w<4; w++) {
            const int tiledRow = TS*t + row;
            const int tiledCol = TS*t + col;
            Asub[row + w*RTS][col] = A[(tiledCol ) + (globalRow+ w*RTS)*K];
            Bsub[row + w*RTS][col] = B[(globalCol ) + (tiledRow+ w*RTS)*N];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k=0; k<TS; k++) {
            for (int w=0; w<4; w++) {
                acc[w] += Asub[row + w*RTS][k] * Bsub[k][col];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w=0; w<4; w++) {
        C[(globalCol) + (globalRow + w*RTS )*N] = acc[w];
    }
    
    printf("TS=%d RTS=%d\n",TS,RTS);
}

