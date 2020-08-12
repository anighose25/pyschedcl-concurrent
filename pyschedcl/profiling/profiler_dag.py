import os
import sys
import subprocess
import json




kernel_info = "./database/info/"

all_kernels = os.listdir(kernel_info)

#print(all_kernels)

lims = [1024,32,8]
GLOBAL_WORK_SIZES = [1024]
local_work_sizes = [4,8,16,32,64,128,256,512]
total_runs = 5
#all_kernels = all_kernels[16:]
#all_kernels = ["FFC.json"]
#print all_kernels
#all_kernels = all_kernels[2:]
#all_kernels= ["FFC.json"]
# all_kernels = all_kernels[3:]
# print all_kernels
all_kernels = [k +".json" for k in ['MatVecMulCoalesced1', 'MatVecMulCoalesced2', 'MatVecMulUncoalesced1', 'atax_kernel1', 'atax_kernel2', 'bicgKernel1', 'covar_kernel', 'fdtd_kernel1', 'fdtd_kernel2', 'fdtd_kernel3', 'gesummv_kernel', 'mean_kernel', 'mm', 'mm2_kernel1', 'mm2metersKernel', 'mm3_kernel1', 'mt', 'mvt_kernel1', 'naive_copy', 'naive_kernel', 'naive_transpose', 'reduce_kernel', 'std_kernel']]
print all_kernels
# sys.exit(-1)
for kernel in all_kernels:
    if kernel.endswith('json') and (not kernel.startswith("Convolution3D_kernel") and not kernel.startswith("gramschmidt")) and (not kernel.startswith("spmv") and (not kernel.startswith("Convolution2D_kernel_")) and (not kernel.startswith("corr_kernel"))):
        print "Profiling ",kernel
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
            for partition in [0,10]:
                for run_number in range(total_runs):
                    dump_file_name = kernel[:-5]+"_"+str(partition)+"_"+str(global_work_size)+"_"+str(local_work_size)+"_"+str(run_number)
                    dump_folder_name = "./profiling/dumps_gtx970/"+kernel[:-5]
                    if not os.path.exists(dump_folder_name):
                        os.makedirs(dump_folder_name)
                    print dump_file_name
                    to_write = "0 {} {}\"dataset\":{},\"n_chunks\":1,\"n_chunks\":1,\"localWorkSize\":{},\"partition\":{}{}\n---\n---\n".\
                    format(kernel,"{",global_work_size,local_work_size,partition,"}")
                    if kernel == "FFC.json" or kernel == "coalesced_gemm.json":
                        to_write = "0 {} {}\"m1\":{},\"p1\":{},\"n1\":{},\"n_chunks\":1,\"wpt\":1,\"localWorkSize\":{},\"TS\":{},\"partition\":{}{}\n---\n---\n".\
                        format(kernel,"{",global_work_size,global_work_size,global_work_size,local_work_size,local_work_size,partition,"}")

                    with open("./dag_info/dag_3_gemm/dag.graph","w") as f:
                        f.write(to_write)
                    print(to_write)
                    subprocess.call("python scheduling/multiple_dag_devices.py -f ./dag_info/dag_3_gemm/ -ng 1 -nc 1 -rc -fdp {}/{}.json".format(dump_folder_name,dump_file_name),shell=True)
    import time
    print "Sleeping for 1 minute"
    time.sleep(60)