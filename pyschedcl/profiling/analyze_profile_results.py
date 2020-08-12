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
from os.path import join
import copy
from os import listdir
from collections import defaultdict
import numpy as np
lims = [1024,64,16]
GLOBAL_WORK_SIZES = [2,64,128,256,512,1024,2048]
local_work_sizes = [1,2,4,8,16,32,64,128]




if __name__ == '__main__':
    profile_folder = "/home/anirban/ResearchTools/pyschedcl-stable/pyschedcl/profiling/dumps/"
    kernel_profile_map = defaultdict(lambda: defaultdict(lambda: defaultdict (lambda: defaultdict(list))))
    kernel_extime_map = defaultdict(lambda: defaultdict (lambda: defaultdict()))
    kernel_class_map = defaultdict(lambda: defaultdict (lambda: defaultdict(str)))
    kernel_cv_map = defaultdict(lambda: defaultdict (lambda: defaultdict(float)))
    for f in listdir(profile_folder):
        if f.endswith('.txt'):
            filename = join(profile_folder,f)
            with open(filename,'r') as g: 
                contents = g.readlines()
        key = f[:-3]
        
        for line in contents:
            name, execution_time = line.strip("\n").split("\t")
            global_work_size,local_work_size,device,run = name.split("_")[-4:] 
            # print key,global_work_size,local_work_size,device

            kernel_profile_map[key][int(global_work_size)][int(local_work_size)][device].append(float(execution_time))

    nCPU = 0
    nGPU = 0
    blacklist = []
    for kernel in kernel_profile_map.keys():
        for global_work_size in kernel_profile_map[kernel].keys():
            if global_work_size != 2:
                for local_work_size in kernel_profile_map[kernel][global_work_size].keys():
                
                    execution_times_cpu = kernel_profile_map[kernel][global_work_size][local_work_size]['CPU']
                    execution_times_gpu = kernel_profile_map[kernel][global_work_size][local_work_size]['GPU']
                
                    if len(execution_times_cpu) > 0 or len(execution_times_gpu) > 0:
                        avg_time_cpu = np.mean(execution_times_cpu)
                        avg_time_gpu = np.mean(execution_times_gpu)
                        std_cpu = np.std(execution_times_cpu)
                        std_gpu = np.std(execution_times_gpu)

                        # print kernel,global_work_size,local_work_size, execution_times_cpu,execution_times_gpu,avg_time_cpu,avg_time_gpu    
                        kernel_extime_map[key][global_work_size][local_work_size] = ((avg_time_cpu,std_cpu),(avg_time_gpu,std_gpu))
                        kernel_cv_map[key][global_work_size][local_work_size] = (std_cpu/avg_time_cpu,std_gpu/avg_time_gpu)
                        if avg_time_cpu <= avg_time_gpu:
                            kernel_class_map[kernel][global_work_size][local_work_size] = "CPU"
                            nCPU +=1
                        else:
                            kernel_class_map[kernel][global_work_size][local_work_size] = "GPU"
                            nGPU +=1
                    else:
                        blacklist.append((kernel,global_work_size,local_work_size))

    import json
    # print json.dumps(kernel_profile_map,indent=2)
    print json.dumps(kernel_class_map, indent=2)
    # print json.dumps(kernel_extime_map, indent=2) 
    print json.dumps(kernel_cv_map, indent=2)
    
    print nCPU, "CPU bound tasks", nGPU, "GPU bound tasks out of a total of ", nCPU+nGPU ,"tasks"
    print "no of kernels missing data", len(blacklist)

