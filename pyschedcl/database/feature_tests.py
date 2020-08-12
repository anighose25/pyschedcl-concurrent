#! /usr/bin/python


import os;
import subprocess;
import time;

# tests = {
# "memset.cl" : ["memset1", "memset2"],
# "mm.cl" : ["mm"],
# "mt.cl" : ["mt"],
# "2DConvolution.cl" : ["Convolution2D_kernel"],
# "2mm.cl" : ["mm2_kernel1"],
# "3DConvolution.cl" : ["Convolution3D_kernel"],
# "3mm.cl" : ["mm3_kernel1"],
# "atax.cl" : ["atax_kernel1", "atax_kernel2"],
# "bicg.cl" : ["bicgKernel1"],
# "correlation.cl" : ["mean_kernel", "std_kernel", "reduce_kernel", "corr_kernel"],
# "covariance.cl" : ["mean_kernel", "reduce_kernel", "covar_kernel"],
# "fdtd2d.cl" : ["fdtd_kernel1", "fdtd_kernel2", "fdtd_kernel3"],
# "gemm.cl" : ["gemm"],
# "gesummv.cl" : ["gesummv_kernel"],
# "gramschmidt.cl" : ["gramschmidt_kernel1", "gramschmidt_kernel2", "gramschmidt_kernel3"],
# "mm2metersKernel.cl" : ["mm2metersKernel"],
# "mvt.cl" : ["mvt_kernel1"],
# "syr2k.cl" : ["syr2k_kernel"],
# "syrk.cl" : ["syrk_kernel"],
# "spmv.cl" : ["spmv_jds_naive"],
# "stencil.cl" : ["naive_kernel"]
# };

tests = {
"gemm.cl" : ["gemm","coalesced_gemm"],
"FFC.cl" : ["FFC"],
"transpose.cl" : ["naive_copy","naive_transpose","coalesced_copy","coalesced_transpose"],
"oclMatVecMul.cl" : ["MatVecMulCoalesced2","MatVecMulCoalesced1","MatVecMulUncoalesced1"],
"softmax.cl" : ["softmax"]
}


def extract_features(test,kernel):
    command="~/Tools/Compilers/llvm-analysis-tools/feature_extraction/coarsening_pass/thrud/tools/scripts/extract_feat.sh "

    input_file = "kernels/"+test
    feature_file ="features/"+kernel+".feat"
    command = command + input_file + " " + kernel + " " + feature_file
    print command
    os.system(command)

if __name__=='__main__':

  for test in tests:
    kernels = tests[test];
    for kernel in kernels:
        extract_features(test, kernel)
