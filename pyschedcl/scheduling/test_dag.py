import sys
import os
sys.path.append(os.getcwd())
import pyschedcl as fw
import logging
import argparse
import json
#import sys
import time
import datetime
#import plotly.plotly as py
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import networkx as nx
import csv
import random
import time
import threading
import numpy as np
logging.basicConfig(level=logging.INFO)
fw.logging.basicConfig(level=logging.INFO)


def parse_arg(args=None):
    parser = argparse.ArgumentParser(description='Schedule set of independent OpenCL Task directed acyclic grpah on CPU-GPU heterogeneous multicores')
    parser.add_argument('-f', '--file',
                        help='Input task file containing list of <json filename, partition class, dataset> tuples',
                        default ='dag_info/dag_transformer/')
    parser.add_argument('-ng', '--nGPU',
                        help='Number of GPUs',
                        default='1')
    parser.add_argument('-nc', '--nCPU',
                        help='Number of CPUs',
                        default='1')
    parser.add_argument('-l', '--log',
                        help='Flag for turning on LOG',
                        action="store_true")
    parser.add_argument('-g', '--graph',
                        help='Flag for plotting GANTT chart for execution',
                        action="store_true")
    parser.add_argument('-df', '--dump_output_file',
                        help='Flag for dumping output file for a kernel',
                        action="store_true")
    parser.add_argument('-t', '--task',
                        help='reduce everything to a single task',
                        action="store_true",
                        default=False)

    parser.add_argument('-ag', '--all_gpu',
                        help='if --task/-t flag is enabled all kernels are moved to gpu if -ag is on, else cpu',
                        action="store_true",
                        default=False)

    parser.add_argument('-rc','--recreate_dag',
                        help = 'recreate the dag json file from dag.graph specification',
                        action = "store_true",
                        default=False
                        )

    parser.add_argument('-nchk','--num_chunks',
                        help = 'number of chunks to split the kernels',
                        default='1'
                        )

    parser.add_argument('-ce','--check_error',
                        help = 'print error for some kernels',
                        action = "store_true",
                        default = False)

    parser.add_argument('-thd','--use_thread',
                        help = "Use threaded scheduler",
                        action = "store_true",
                        default = False)

    parser.add_argument('-fdp','--full_dump_path',
                        help = "Specify Full Dump Path for profiling results",
                        default='None')

    return parser.parse_args(args)

def all_done():
    for dag in all_dags:
        if not dag.finished():
            return False
    return True



def random_selector(Q,start_sched):
    fw.frontier_Q_lock.acquire()
    while ((not Q) and (not all_done())):
        fw.frontier_Q_lock.wait()

    if all_done():
        total_time_in_multiple_dag_devices = time.time()-start_sched
        print "\t \t Total Time measured by the scheduler - ",total_time_in_multiple_dag_devices
        fw.frontier_Q_lock.release()
        return -1
        #fw.frontier_Q_lock.release()


    task = Q[0]
    del Q[0]
    fw.frontier_Q_lock.release()
    return task


if __name__ == '__main__':
    args = parse_arg(sys.argv[1:])

    if args.recreate_dag:
        fw.create_dag("./database/info/","./dag_info/dag_transformer/dag.graph","./dag_info/dag_transformer/t1.json"\
    ,partition=10)

    num_chunks = int(args.num_chunks)
    fw.just_for_testing_num_chunks = num_chunks


    info_file = args.file
    cmd_qs, ctxs, gpus, cpus = fw.host_initialize(int(args.nGPU), int(args.nCPU),use_mul_queues = True)
    print "command Queue -"
    print cmd_qs
    print "ready Queue -"
    print fw.ready_queue
    print(gpus,cpus)


    #Dags_folder = list()
    all_dags = [] #list of all the DAGs

    finished_task_Dag = dict()
    deleted_task_dag = list()

    all_dags_jsons = [join(info_file,f) for f in listdir(info_file)] #list of json files - each json file corresponds to a single DAG
    gantt_label = [(info_file + f) for f in listdir(info_file)]
    gantt = 0
    # count  = 0
    # count1 = 0
    # task_dag_id = 0
    frontier_Q = fw.frontier_Q
    ex_stats ="logs/transformer_profiling_128_128_128.json"
    for i,dag_json_file in  enumerate(all_dags_jsons):
        if dag_json_file.endswith('json'):
            logging.debug("main : Reading json file "+ dag_json_file)
            with open(dag_json_file,"r") as f:
                info = json.loads(f.read())
            logging.debug("main : prepraing task dag number "+str(i))
            all_dags.append(fw.TaskDAG(info,dag_number = i ,dataset = 1024,map_all=args.task,all_map_to_gpu=args.all_gpu,\
            gpus=gpus,cpus=cpus,ctxs=ctxs,cmd_qs=cmd_qs,ex_stats_file=ex_stats)) 
            logging.debug("main : prepared task dag number "+str(i)+"\n\n")
            fw.frontier_Q_lock.acquire()
            frontier_Q.extend(all_dags[-1].free_tasks)
            for task in all_dags[-1].free_tasks:
                task.has_enqueued = True
            fw.frontier_Q_lock.release()


    dag = all_dags[0]
   
    dag.print_task_information()
    tasks = dag.G.nodes()
    for i in range(len(tasks)):
        print i,tasks[i].get_kernel_ids(),dag.get_task_children_kernel_ids(tasks[i])
    print dag.print_kernel_information()
    dag.compute_blevel_ranks()
    levels= dag.make_levels()
    task_device_bias_map = dag.run_static_heft()
    print task_device_bias_map
    dag.dump_device_biases("logs/heft_bias.png", task_device_bias_map)
    # task_device_bias_map = dag.obtain_task_clusters(levels)
    # dag.dump_device_biases("logs/bias.png", task_device_bias_map)
    # dag.dump_graph("logs/ranked_dag.png",rank=True)
    '''
    dag.dump_graph("logs/test.png")
    
    print "after merging"
    tasks = dag.G.nodes()
    for i in range(len(tasks)):
        if dag.get_task_children(tasks[i]):
            print "merging",tasks[i].get_kernel_ids(),dag.get_task_children_kernel_ids(tasks[i])[0]
            dag.merge_tasks(tasks[i], dag.get_task_children(tasks[i])[0], 10)
            break        
    # dag.merge_tasks(dag.G.nodes()[0], dag.G.nodes()[1], 10)
    dag.dump_graph("logs/test_clustered.png")
 '''