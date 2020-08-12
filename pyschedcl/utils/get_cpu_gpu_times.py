#!/usr/bin/env python

import json, sys, subprocess, os, datetime
import pyschedcl as fw
import argparse
import time
import sys
from decimal import *
# logging.basicConfig(level=logging.DEBUG)
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


def parse_arg(args=None):
    parser = argparse.ArgumentParser(description='Get Profiling Times of Kernels for  CPU and GPU devices')
    

    parser.add_argument('-d', '--dataset_size',
                        help='Inputs the size of the dataset',
                        default='1024')

    parser.add_argument('-nr', '--runs',
                        help='Number of runs for executing each partitioned variant of the original program',
                        default=5)
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_arg(sys.argv[1:])


    DEV_NULL = open(os.devnull, 'w')

    p_path = fw.SOURCE_DIR + 'partition/partition.py'
    i_path = fw.SOURCE_DIR + 'info/'
    span_times = []
   
    dump_file = open("kernel_cpu_gpu_timings_2048.stats" ,'w')
    kernels = ['bicg2.json', 'VectorAdd.json', 'covar.json', 'corr.json', 'gemm.json', 'mm22.json', 'MatVecMulCoalesced1.json', 'atax1.json', 'MatVecMulCoalesced2.json', 'uncoalesced_copy.json']
    for kernel_name in kernels:
        print "Calculating for ", kernel_name
        cpu_time = 0.0
        gpu_time = 0.0
        kernel_execution_stats = kernel_name[:-5] + "_" + str(args.dataset_size) + ":"  
        for p in [0, 10]:

            execute_cmd = "python " + p_path + " -f " + i_path + kernel_name + " -p " + str(p) + " -d " + args.dataset_size + " > temp.log"

            count = 0
            span = 0
            for r in range(int(args.runs)):
                # print execute_cmd
                sys.stdout.write('\r')

                os.system(execute_cmd)
                sys.stdout.flush()

                time.sleep(1)
                get_span_cmd = "cat temp.log |grep span_time"
                # print get_span_cmd
                span_info = os.popen(get_span_cmd).read().strip("\n")
                if "span_time" in span_info:
                    count = count + 1

                    profile_time = float(span_info.split(" ")[1])
                    if profile_time > 0:
                        span = span + profile_time
            avg_span = span/count
            if p == 0:
                kernel_execution_stats += str(avg_span)+","
            else:
                kernel_execution_stats += str(avg_span)+"\n"

       
        print "Dumping ",kernel_execution_stats
        dump_file.write(kernel_execution_stats)
        os.system("rm temp.log")
    # print span_times
    dump_file.close()
