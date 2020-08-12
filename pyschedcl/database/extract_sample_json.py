#! /usr/bin/python


import os;
import subprocess;
import time;

tests = {}
for i in range(1,1001):
    tests["sample_"+str(i)+".cl"]=["A"] 

def create_var_argument(arg_index,datatype):
    value="dataset"
    if datatype == "float":
        value="1.0"
    else:
        value="dataset"
    arg_info = {"pos":arg_index,"value":value,"type":datatype}
    return arg_info


def create_buffer_argument(arg_index,datatype,dataset):
    if dataset == "none":
        dataset="dataset"
    buffer_info = {"break":0,"pos":arg_index,"size":dataset,"type":datatype}
    return buffer_info

def get_work_dimension(test):
    source = open("kernels/"+test).readlines()
    src =""
    for s in source:
        src += s
    dim1_g = src.find("get_global_id(0)")
    dim1_l = src.find("get_local_id(0)")
    dim1_w = src.find("get_group_id(0)")
    dim2_g = src.find("get_global_id(1)")
    dim2_l = src.find("get_local_id(1)")
    dim2_w = src.find("get_group_id(1)")
    dim3_g = src.find("get_global_id(2)")
    dim3_l = src.find("get_local_id(2)")
    dim3_w = src.find("get_group_id(2)")

    dim1 = False
    dim2 = False
    dim3 = False

    if dim1_g >=0 or dim1_l >=0 or dim1_w >=0:
        dim1 = True

    if dim2_g >=0 or dim2_l >=0 or dim2_w >=0:
        dim2 = True

    if dim3_g >=0 or dim3_l >=0 or dim3_w >=0:
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
    kernel = test[:-2]
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
    kernel = test[:-3]
    f = open("features/"+kernel+".feat")
    contents = f.readlines()
    info = []
    info_json = {"inputBuffers": [], "outputBuffers" : [],"ioBuffers": [], "name":"A", "src":test, "varArguments":[]}
    dimension = get_work_dimension(test)
    info_json["workDimension"]=dimension
    info_json["globalWorkSize"]=get_global_worksize(dimension)
    for line in contents:
        if line == "---\n":
            break
        info.append(line.strip("\n"))
    for line in info:
        if "->" in line:
            arg, stat = line.split("->")
            arg = arg.strip(" ")
            stat = stat.strip(" ")
            inf,buf = stat.split(",")
            inf = inf.split("_")
            if len(inf)>1:
                if inf[1]=='input':
                    info_json["inputBuffers"].append(create_buffer_argument(int(arg),inf[0],buf))
                elif inf[1]=='output':
                    info_json["outputBuffers"].append(create_buffer_argument(int(arg),inf[0],buf))
                else:
                    info_json["ioBuffers"].append(create_buffer_argument(int(arg),inf[0],buf))
            else:
                info_json["varArguments"].append(create_var_argument(int(arg),inf[0]))
    info_json_file = "info/"+kernel+".json"
    import json
    with open(info_json_file, 'w') as op:
        json.dump(info_json, op,indent=2)

def extract_features(test,kernel):
    command="~/ResearchTools/Github/pyschedcl-stable/pyschedcl/passes/feature_pass/thrud/tools/scripts/extract_feat.sh "
    
    input_file = "kernels/"+test
    feature_file ="features/"+test[:-3]+".feat"
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

