#!/usr/bin/env python
import sys
import os
sys.path.append(os.getcwd())


import json
import sys
import threading
import datetime
import time
import pyschedcl as fw
import logging
import argparse

#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
logging.basicConfig(level=logging.DEBUG)
fw.logging.basicConfig(level=logging.DEBUG)

def parse_arg(args=None):
    parser = argparse.ArgumentParser(description='Partition kernels into CPU and GPU heterogeneous system')
    parser.add_argument('-f', '--file',
                        help='Input the json file',
                        required='True')
    parser.add_argument('-p', '--partition_class',
                        help='Inputs the partition ratio')
    parser.add_argument('-d', '--dataset_size',
                        help='Inputs the size of the dataset',
                        default='512')
    parser.add_argument('-ng', '--nGPU',
                        help='Number of GPUs',
                        default='4')
    parser.add_argument('-nc', '--nCPU',
                        help='Number of CPUs',
                        default='2')
    parser.add_argument('-l', '--log',
                        help='Flag for turning on LOG',
                        action="store_true")
    parser.add_argument('-g', '--graph',
                        help='Flag for plotting GANTT chart for execution',
                        action="store_true")
    parser.add_argument('-df', '--dump_output_file',
                        help='Flag for dumping output file for a kernel',
                        action="store_true")

    parser.add_argument('-cd', '--custom_data',
                        help='Flag for generating custom data for vecAdd (hardcoded)',
                        action="store_true")

    parser.add_argument('-pf', '--profile',
                        help='Flag for dumping the profile',
                        action="store_true")

    return parser.parse_args(args)


def test(kernel_name,ref_ip,ref_op,dataset):
    print kernel_name
    if kernel_name == "mm":
        print "testing matrix multiplication"
        i1 = ref_ip[0].reshape([dataset,-1])
        i2 = ref_ip[1].reshape([dataset,-1])
        o_pred = ref_op[0].reshape([dataset,-1])
        o_act = i1.dot(i2)
        print o_pred
        print o_act
        return o_pred == o_act

if __name__ == "__main__":
    args=parse_arg(sys.argv[1:])
    src_name = args.file.split("/")[-1]
    s_name = src_name[:-5]    #name of Kernel
    if args.log:
        logging.basicConfig(level=logging.DEBUG)
    cmd_qs,ctxs, gpus, cpus = fw.host_initialize(int(args.nGPU), int(args.nCPU))
    info_file = args.file
    with open(info_file,"r") as f:
        info = json.loads(f.read())
    dataset = int(args.dataset_size)

    st = time.time()
    if args.dump_output_file:
        fw.dump_output = True
    if args.partition_class != None :
        partition = int(args.partition_class)
        kernel = fw.Kernel(info, dataset=dataset, partition=partition)
    else:
        kernel = fw.Kernel(info, dataset=dataset)
        partition = info['partition']

    name = s_name + '_' + str(partition) + '_' + str(dataset) + '_' + str(time.time()).replace(".","")
    if args.log:
        f_path = fw.SOURCE_DIR + 'logs/' + name + '_debug.log'
        logging.basicConfig(filename=f_path, level=logging.DEBUG)
        print "LOG file is saved at %s" % f_path
    logging.debug('Number of GPUs : %d' % (len(gpus)))
    logging.debug(gpus)
    logging.debug('Number of CPUs : %d' % (len(cpus)))
    logging.debug(cpus)
    sched_start_time = datetime.datetime.now()
    logging.debug("Building Kernel...")
    kernel.build_kernel(gpus, cpus, ctxs)
    logging.debug("Loading Kernel Data...")
    if not args.custom_data:
        kernel.random_data()
    else:
        #hardcoded for vecAdd
        import numpy as np
        a = np.random.randn(dataset,)
        b = np.random.randn(dataset,)
        data = {"input" : [a,b]}
        kernel.load_data(data)
        del data,a,b

    import weakref
    ref_ip = []
    ref_op = []
    for ip in kernel.data["input"]:
        ref_ip.append(weakref.proxy(ip))

    for op in kernel.data["output"]:
        ref_op.append(weakref.proxy(op))

    kernel.get_data_types_and_shapes()


    logging.debug("Dispatching Kernel...")

    lw_callback = False



    start_time, done_events = kernel.dispatch(0, 0, ctxs, cmd_qs,C_cpu=1,C_gpu=1)



    logging.debug("Waiting for events... \n")
    fw.host_synchronize(cmd_qs, done_events)



    sched_end_time = datetime.datetime.now()
    seconds = (sched_end_time - sched_start_time).total_seconds()
    # print("%s with partition %d and dataset %d ran %fs" % (info['name'], partition, dataset, seconds))


    if args.profile:
        dump = os.path.join("profiling","dumps")
        if not os.path.exists(dump):
            os.makedirs(dump)
        with open(os.path.join(dump,name),"w") as f:
            f.write(str(seconds))

    if not lw_callback:
        dump_dev=fw.dump_device_history()

    if args.graph:
        filename = fw.SOURCE_DIR + 'gantt_charts/' + name + '.png'
        fw.plot_gantt_chart_graph(dump_dev, filename)

    ##testing if kernel can release host arrays
    ##input ->
    for i,ip in enumerate(ref_ip):
        print "input number ",i+1
        print ip
        print "\n"

    ##output ->
    for i,op in enumerate(ref_op):
        print "output number ",i+1
        print op
        print "\n"

    print test(s_name,ref_ip,ref_op,dataset)


    kernel.release_host_arrays()

    for ref in ref_ip:
        try:
            print "after deletion :- ",ref
        except ReferenceError:
            print("Host input array successfully deleted")

    en = time.time()

    print "time taken, ",en-st
