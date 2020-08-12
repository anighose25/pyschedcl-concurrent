#include "opencl-shim.h"
import os
kernels=os.listdir("./")
counter = 1
for kernel in kernels:
	os.rename(kernel,"sample_"+str(counter)+".cl")
	counter +=1
