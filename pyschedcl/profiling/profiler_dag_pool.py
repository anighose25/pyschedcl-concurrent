import os
import sys
import subprocess
import json




kernel_info = "./database/info/"

all_kernels = os.listdir(kernel_info)

batch = 32
channel = 96
in_height = 55
in_width = 55
out_height = 27
out_width = 27

inputSize = batch*channel*in_height*in_width
outputSize = batch*channel*out_height*out_width

lims = [1024,32,8]
GLOBAL_WORK_SIZES = [outputSize]
local_work_sizes = [4,8,16,32,64,128,256,512]
total_runs = 5


all_kernels = [k +".json" for k in ["expanded_pooling_layers"]]
print(all_kernels)
# sys.exit(-1)
for kernel in all_kernels:
    if kernel.endswith('json') and (not kernel.startswith("Convolution3D_kernel") and not kernel.startswith("gramschmidt")) and (not kernel.startswith("spmv") and (not kernel.startswith("Convolution2D_kernel_")) and (not kernel.startswith("corr_kernel"))):
        print("Profiling ",kernel)
        with open("./database/info/"+kernel) as f:
            json_source = json.load(f)
        #print json.dumps(json_source,indent=2)
        work_dimension = json_source["workDimension"]
        limit_of_local_work_size = lims[work_dimension-1]
        #sys.exit(-1)

    else:
        continue
    for global_work_size in GLOBAL_WORK_SIZES:
        for local_work_size in local_work_sizes:
            if local_work_size > global_work_size:
                break
            if local_work_size > limit_of_local_work_size:
                break
            if work_dimension <=2 and local_work_size <= lims[work_dimension]:
                continue
            for partition in [0]:
                for run_number in range(total_runs):
                    dump_file_name = kernel[:-5]+"_"+str(partition)+"_"+str(global_work_size)+"_"+str(local_work_size)+"_"+str(run_number)
                    dump_folder_name = "./profiling/dumps_gtx970/"+kernel[:-5]
                    if not os.path.exists(dump_folder_name):
                        os.makedirs(dump_folder_name)
                    print(dump_file_name)
                    # to_write = "0 {} {}\"dataset\":{},\"inputSize\":{},\"outputSize\":{},\"localWorkSize\":{},\"partition\":{}{}\n---\n---\n".\
                    # format(kernel,"{",global_work_size, inputSize, outputSize, local_work_size, partition,"}")

                    to_write = "0 {} {}\"channel\":{},\"height\":{},\"width\":{},\"inputSize\":{},\"outputSize\":{},\"partition\":{}{}\n---\n---\n".\
                    format(kernel,"{", channel, out_height, out_width, inputSize, outputSize, partition,"}")
                    
                    with open("./dag_info/dag_3_gemm/dag.graph","w") as f:
                        f.write(to_write)
                    print(to_write)
                    subprocess.call("python scheduling/multiple_dag_devices.py -f ./dag_info/dag_3_gemm/ -ng 0 -nc 1 -rc -fdp {}/{}.json".format(dump_folder_name,dump_file_name),shell=True)
    import time
    print("Sleeping for 1 minute")
    time.sleep(60)
