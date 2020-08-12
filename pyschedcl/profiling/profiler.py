import sys
import os
sys.path.append(os.getcwd())
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import json
import sys
import threading
import datetime
import time
import pyschedcl as fw
import logging
import argparse
from tests import test
import weakref
import importlib

import copy


'''
1D:  ['gramschmidt_kernel1(done)'
'gesummv_kernel(done)', 'atax_kernel1'(done), 'atax_kernel2'(done), 'spmv_jds_naive', 'mvt_kernel1(done)']


2D:  ['gemm'(done), 'mean_kernel(done)', 'std_kernel(done)', 'reduce_kernel(done)', 'corr_kernel(done)',
'fdtd_kernel1(done)', 'fdtd_kernel2(done)', 'fdtd_kernel3(done)', 'memset1(done)', 'memset2(done)',
'mm2_kernel1(done)', 'mm'(done), 'mm2metersKernel(done)', 'syrk_kernel(done)', 'mm3_kernel1(done)', 'syr2k_kernel(done)', 'mt'(done),
'Convolution3D_kernel(done)', 'Convolution2D_kernel(done)']

3D:  ['naive_kernel'(done)]

'''

#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'



def parse_arg(args=None):
    parser = argparse.ArgumentParser(description='Partition kernels into CPU and GPU heterogeneous system')
    parser.add_argument('-f', '--file',
                        help='Input the json file',
                        required='True')

    parser.add_argument('-d', '--dataset_size',
                        help='Inputs the size of the dataset',
                        default='512')


    parser.add_argument('-p', '--partition_class',
                        help='Inputs the partition ratio')



    return parser.parse_args(args)



lims = [1024,32,16]
GLOBAL_WORK_SIZES = [32,64,128,256,512,1024,2048]
local_work_sizes = [16,32,64,128]



if __name__ == '__main__':

    args=parse_arg(sys.argv[1:])
    info_file = args.file
    src_name = info_file.split("/")[-1]
    kernel_name = src_name[:-5]    #name of Kernel

    ## Request devices from the system


    print info_file
    #logging.info("Kernel :- "+kernel_name)
    logging.basicConfig(level=logging.INFO,filename="/home/anirban/ResearchTools\
/pyschedcl-siddharth/pyschedcl-stable/pyschedcl/profiling/dumps_compute_0_0/log_"+str(kernel_name)+".txt",filemode='wa+')

    for dataset in GLOBAL_WORK_SIZES:
        #logging.info("\tGlobal Work Size :- "+str(dataset))
        #local_work_size = 512
        for local_work_size in local_work_sizes:
            if local_work_size > dataset:
                continue
            #logging.info("\t\tLocal Work Size :- "+str(local_work_size))
            print "dataset size :- ",dataset
            for partition in [0,10]:
                with open(info_file,"r") as f:
                    info = json.loads(f.read())
                    info['localWorkSize'] = [local_work_size]*info['workDimension']

                if local_work_size >= lims[info['workDimension']-1]:
                    break
                elif info['workDimension'] == 3 and dataset > 256:
                    break


                cmd_qs,ctxs, gpus, cpus = fw.host_initialize(1, 1)
                if partition == 0:
                    dev = "CPU"
                else:
                    dev = "GPU"
                print dev

                kernel = fw.Kernel(info, dataset=dataset, partition=partition)
                kernel.build_kernel(gpus, cpus, ctxs,profile=True)
                kernel.random_data()


                ref_ip = []
                ref_op = []
                ref_iop = []
                for i,ip in enumerate(kernel.data["input"]):
                    ref_ip.append(weakref.proxy(ip))
                    print "input number ",i, ip.shape

                for i,op in enumerate(kernel.data["output"]):
                    ref_op.append(weakref.proxy(op))
                    print "output number ",i, op.shape


                for i,iop in enumerate(kernel.data["io"]):
                    ref_iop.append(weakref.proxy(iop))
                    print "input output number ",i, iop.shape

                times = []
                for trial_number in range(20):
                    sched_start_time = datetime.datetime.now()
                    start_time, done_events = kernel.dispatch(0, 0, ctxs, cmd_qs,C_cpu=1,C_gpu=1)
                    fw.host_synchronize(cmd_qs, done_events)
                    sched_end_time = datetime.datetime.now()
                    seconds = (sched_end_time - sched_start_time).total_seconds()
                    logging.info(kernel_name+"_"+str(dataset)+"_"+str(local_work_size)+"_"+str(dev)+"_"+str(trial_number)+"\t"+str(seconds))
                    time.sleep(0.1)
                    times.append(seconds)

                print "time taken :- ",seconds,"\n"

                #logging.info("\n\n")


                #print test(kernel_name,ref_ip,ref_op,ref_iop,dataset)
                kernel.release_host_arrays()


            local_work_size*=2
