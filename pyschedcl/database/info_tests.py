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

def create_var_argument(arg_index,datatype):
    arg_info = {"pos":arg_index,"value":"dataset","type":datatype}
    return arg_info


def create_buffer_argument(arg_index,datatype):
    buffer_info = {"break":0,"pos":arg_index,"size":"dataset","type":datatype}
    return buffer_info

def get_work_dimension(test):
    source = open("kernels/"+test).readlines()
    src =""
    for s in source:
        src += s
    dim1_g = src.find("get_global_id(0)")
    dim1_l = src.find("get_local_id(0)")
    dim2_g = src.find("get_global_id(1)")
    dim2_l = src.find("get_local_id(1)")
    dim3_g = src.find("get_global_id(2)")
    dim3_l = src.find("get_local_id(2)")

    dim1 = False
    dim2 = False
    dim3 = False

    if dim1_g >=0 or dim1_l >=0:
        dim1 = True

    if dim2_g >=0 or dim2_l >=0:
        dim2 = True

    if dim3_g >=0 or dim3_l >=0:
        dim3 = True

    if dim3:
        return 3
    if dim2:
        return 2
    if dim1:
        return 1

def get_global_worksize(dimension):
    if dimension == 1:
        return "[dataset]"
    if dimension == 2:
        return "[dataset,dataset]"
    if dimension == 3:
        return "[dataset,dataset,dataset]"


def feature_map(test,kernel):

    f = open("features/"+kernel+".feat")
    contents = f.readlines()
    features = {}
    counter = 0
    for line in contents:
        
        counter += 1
        if line == "---\n":
            break
    contents=contents[counter:][:-7]
    print contents
    for line in contents:
        print line
        feature_name,feature_value = line.strip("\n").strip("\t").split(":")
        features[feature_name]=int(feature_value)
    print features

def dump_json(test,kernel):
    f = open("features/"+kernel+".feat")
    contents = f.readlines()
    info = []
    info_json = {"inputBuffers": [], "outputBuffers" : [],"ioBuffers": [], "name":kernel, "src":test, "varArguments":[]}
    dimension = get_work_dimension(test)
    info_json["workDimension"]=dimension
    info_json["globalWorkSize"]=get_global_worksize(dimension)
    for line in contents:
        if line == "---\n":
            break
        info.append(line.strip("\n"))
    for line in info:
        arg, inf = line.split("->")
        arg = arg.strip(" ")
        inf = inf.strip(" ")
        inf = inf.split("_")
        if len(inf)>1:
            if inf[1]=='input':
                info_json["inputBuffers"].append(create_buffer_argument(int(arg),inf[0]))
            elif inf[1]=='output':
                info_json["outputBuffers"].append(create_buffer_argument(int(arg),inf[0]))
            else:
                info_json["ioBuffers"].append(create_buffer_argument(int(arg),inf[0]))
        else:
            info_json["varArguments"].append(create_var_argument(int(arg),inf[0]))
    info_json_file = "info/"+kernel+".json"
    import json
    with open(info_json_file, 'w') as op:
        json.dump(info_json, op,indent=2)

def extract_features(test,kernel):
    command="~/Tools/Compilers/llvm-analysis-tools/feature_extraction/coarsening_pass/thrud/tools/scripts/extract_feat.sh "
    
    input_file = "kernels/"+test
    feature_file ="features/"+kernel+".feat"
    command = command + input_file + " " + kernel + " " + feature_file
    print command
    os.system(command)

if __name__=='__main__':
    #dump_json("mm.cl","mm")
    #feature_map("mm.cl","mm")

  for test in tests:
    kernels = tests[test];
    for kernel in kernels:
        dump_json(test, kernel)   

