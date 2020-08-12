#! /usr/bin/python


import os;
import subprocess;
import time;

tests = {
"memset.cl" : ["memset1", "memset2"], 
"mm.cl" : ["mm"], 
"mt.cl" : ["mt"],
"2DConvolution.cl" : ["Convolution2D_kernel"],
"2mm.cl" : ["mm2_kernel1"],
"3DConvolution.cl" : ["Convolution3D_kernel"],
"3mm.cl" : ["mm3_kernel1"],
"atax.cl" : ["atax_kernel1", "atax_kernel2"],
"bicg.cl" : ["bicgKernel1"],
"correlation.cl" : ["mean_kernel", "std_kernel", "reduce_kernel", "corr_kernel"],
"covariance.cl" : ["mean_kernel", "reduce_kernel", "covar_kernel"],
"fdtd2d.cl" : ["fdtd_kernel1", "fdtd_kernel2", "fdtd_kernel3"],
"gemm.cl" : ["gemm"],
"gesummv.cl" : ["gesummv_kernel"],
"gramschmidt.cl" : ["gramschmidt_kernel1", "gramschmidt_kernel2", "gramschmidt_kernel3"],
"mm2metersKernel.cl" : ["mm2metersKernel"],
"mvt.cl" : ["mvt_kernel1"],
"syr2k.cl" : ["syr2k_kernel"],
"syrk.cl" : ["syrk_kernel"],
"spmv.cl" : ["spmv_jds_naive"],
"stencil.cl" : ["naive_kernel"]
}; 




if __name__=='__main__':
    #dump_json("mm.cl","mm")
    import json
    kernels_1D = []
    kernels_2D = []
    kernels_3D = []
    for test in tests:
        kernels = tests[test];
        for kernel in kernels:
            info = json.loads(open("info/"+kernel+".json").read())   
            if info["workDimension"]==1:
                kernels_1D.append(kernel)
            elif info["workDimension"]==2:
                kernels_2D.append(kernel)
            else:
                kernels_3D.append(kernel)
    print "1D: ", kernels_1D
    print "2D: ", kernels_2D
    print "3D: ", kernels_3D

    
    print "(",
    for kernel in kernels_1D+kernels_2D+kernels_3D:
        print "\""+kernel+"\" ",
    print ")"
