#define SCORE(i, j) input_itemsets_l[j + i * (BLOCK_SIZE+1)]
#define REF(i, j)   reference_l[j + i * BLOCK_SIZE]


int maximum( int a,
		 int b,
		 int c){

	int k;
	if( a <= b )
		k = b;
	else 
	k = a;

	if( k <=c )
	return(c);
	else
	return(k);
}

__kernel void 
nw_kernel(__global int  * reference_d,            //0 input - BS+1 x BS+1
		   __global int  * input_itemsets_d,      //1 io - BS+1 x BS+1
           __global int  * upper_row_d,           //2 io - BS 
           __global int  * left_col_d,            //3 io - BS 
           __global int * top_left_d,             //4 io - 1
		   __local	int  * input_itemsets_l,      //5 local - (BS+1 x BS+1)
		   __local	int  * reference_l,           //6 local - BSxBS
           int cols,                              //7 BS+1
           int penalty,                           //8 10
           int blk,                               //9 1 - fixed
           int block_width,                       //10 not needed
           int worksize,                          //11 BS
           int offset_r,                          //12 0
           int offset_c                           //13 0
    )
{  

	// Block index
    int bx = get_group_id(0);	                   //always 0
	//int bx = get_global_id(0)/BLOCK_SIZE;
   
    // Thread index
    int tx = get_local_id(0);                      //[0,BS-1]
    
    // Base elements
    int base = offset_r * cols + offset_c;         //0
    
    int b_index_x = bx;                            //0
	int b_index_y = blk - 1 - bx;                  //0
	
	
    //each thread handles a column
    //index - position of topmost element of the column 
    //index_n - position of the element above index 
    //index_w - position of element 
	int index   =   tx + ( cols + 1 );
	int index_n   = tx;
	int index_w   = 0;
	int index_nw =  0;
   
    
	if (tx == 0){
		SCORE(tx, 0) = top_left_d[0];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
		REF(ty, tx) =  reference_d[index + cols * ty];

	barrier(CLK_LOCAL_MEM_FENCE);

	SCORE((tx + 1), 0) = left_col_d[index_w + tx]; //copying right column into local memory

	barrier(CLK_LOCAL_MEM_FENCE);

	SCORE(0, (tx + 1)) = upper_row_d[index_n]; //copying upper row into local memory
  
	barrier(CLK_LOCAL_MEM_FENCE);
	
	
	for( int m = 0 ; m < BLOCK_SIZE ; m++){
	
	  if ( tx <= m ){
	  
		  int t_index_x =  tx + 1;
		  int t_index_y =  m - tx + 1;
			
		  SCORE(t_index_y, t_index_x) = maximum( SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1)),
		                                         SCORE((t_index_y),   (t_index_x-1)) - (penalty), 
												 SCORE((t_index_y-1), (t_index_x))   - (penalty));
	  }
	  barrier(CLK_LOCAL_MEM_FENCE);
    }
    
     barrier(CLK_LOCAL_MEM_FENCE);
    
	for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){
   
	  if ( tx <= m){
 
		  int t_index_x =  tx + BLOCK_SIZE - m ;
		  int t_index_y =  BLOCK_SIZE - tx;

         SCORE(t_index_y, t_index_x) = maximum(  SCORE((t_index_y-1), (t_index_x-1)) + REF((t_index_y-1), (t_index_x-1)),
		                                         SCORE((t_index_y),   (t_index_x-1)) - (penalty), 
		 										 SCORE((t_index_y-1), (t_index_x))   - (penalty));
	   
	  }

	  barrier(CLK_LOCAL_MEM_FENCE);
	}
	

   for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
     input_itemsets_d[index + cols * ty] = SCORE((ty+1), (tx+1));

    upper_row_d[tx] = SCORE((BLOCK_SIZE), (tx+1));
    left_col_d[tx] = SCORE((tx+1), (BLOCK_SIZE));
    
    if (tx == 0){
		top_left_d[0] = SCORE(tx, 0);
	}

	barrier(CLK_LOCAL_MEM_FENCE);



    return;
   
}