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

# logging.basicConfig(level=logging.CRITICAL)
# fw.logging.basicConfig(level=logging.CRITICAL)

# logging.basicConfig(level=logging.DEBUG)
# fw.logging.basicConfig(level=logging.DEBUG)


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
        print("\t \t Total Time measured by the scheduler - ",total_time_in_multiple_dag_devices)
        fw.frontier_Q_lock.release()
        return -1
        #fw.frontier_Q_lock.release()


    task = Q[0]
    del Q[0]
    fw.frontier_Q_lock.release()
    return task


if __name__ == '__main__':
    args = parse_arg(sys.argv[1:])
    file_prefix = args.file
    if args.recreate_dag:
        fw.create_dag("./database/info",file_prefix + "dag.graph",file_prefix + "t1.json", partition=10)
    #     fw.create_dag("./dag_info/dag_3_gemm/info","./dag_info/dag_3_gemm/dag.graph","./dag_info/dag_3_gemm/t1.json"\
    # ,partition=10)

    num_chunks = int(args.num_chunks)
    fw.just_for_testing_num_chunks = num_chunks


    info_file = args.file
    print(info_file)

    cmd_qs, ctxs, gpus, cpus = fw.host_initialize(int(args.nGPU), int(args.nCPU),use_mul_queues = True)


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

    for i,dag_json_file in  enumerate(all_dags_jsons):
        if dag_json_file.endswith('json'):
            logging.debug("main : Reading json file "+ dag_json_file)
            with open(dag_json_file,"r") as f:
                info = json.loads(f.read())
            logging.debug("main : prepraing task dag number "+str(i))
            all_dags.append(fw.TaskDAG(info,dag_number = i ,dataset = 1024,map_all=args.task,all_map_to_gpu=args.all_gpu,\
            gpus=gpus,cpus=cpus,ctxs=ctxs,cmd_qs=cmd_qs,use_predefined_mapping=True)) #create dag for info file (ex :- dag_test1/t1.json)
            logging.debug("main : prepared task dag number "+str(i)+"\n\n")
            fw.frontier_Q_lock.acquire()
            frontier_Q.extend(all_dags[-1].free_tasks)
            for task in all_dags[-1].free_tasks:
                task.has_enqueued = True
            fw.frontier_Q_lock.release()


    logging.debug("printing initial frontier_Q tasks\n\n")
    # for i,task in enumerate(frontier_Q):
    #     logging.debug("task number "+str(i+1)+ " "+ task.id)
    #     logging.debug("it's free kernels "+str([k.id for k in task.free_kernels]))
    #     logging.debug("it's all kernels "+str([k.id for k in task.kernels]))
    #     logging.debug("it's dag id "+str(task.task_dag_object.id))
    #     logging.debug("it's optm device is "+str(task.optm_device))

    #sys.exit(-1)
    start_sched = time.time()
    while True:
        logging.debug("before selection length of frontier_Q : "+str(len(frontier_Q)))
        next_task = random_selector(frontier_Q,start_sched)
        if next_task == -1:
            logging.debug("all dags are finished ")
            break
        logging.debug("task selected "+str(next_task.id))
        optm_device = next_task.optm_device
        if int(optm_device) == 10:
            optm_device = "gpu"
            logging.debug("gpu selected")
        elif int(optm_device) == 0:
            optm_device = "cpu"
            logging.debug("cpu selected")
        else:
            raise

        logging.debug("after selection length of frontier_Q : "+str(len(frontier_Q))+"\n")



        fw.rqLock.acquire()
        while not (len(fw.ready_queue[optm_device]) > 0):
            fw.rqLock.wait()


        #now device is free and time to schedule task
        logging.debug("current ready queue "+str(fw.ready_queue[optm_device]))
        #print(list(fw.ready_queue[optm_device]))

        next_task.allocate_devices(list(fw.ready_queue[optm_device]))
        logging.debug(str(fw.ready_queue[optm_device]))
        fw.rqLock.release()


        if args.use_thread:
            dispatch_thread = threading.Thread(target=next_task.dispatch_all,args=())
            dispatch_thread.start()

        else:
            next_task.dispatch_all(gpus, cpus, ctxs, cmd_qs)
        #next_task.dispatch_all(gpus, cpus, ctxs,cmd_qs)
        #fw.rqLock.acquire()
        #logging.debug("Number of threads running are "+str(threading.active_count()))



    logging.debug("Profiling the execution")

    ref = {'cpu' : None, 'gpu' : None}


    for dag in all_dags:
        for kernel_id,kernel in dag.kernels.items():
            #print "\t Kernel ",kernel.name
            dev = kernel.task_object.device
            for ev in kernel.write_events:
                ev.wait()
                start_time = ev.profile.START *1e-9
                end_time =   ev.profile.END *1e-9
                if not ref[dev]:
                    ref[dev] = start_time
                else:
                    ref[dev] = min(ref[dev],start_time)

            for ev in kernel.nd_range_event:
                ev.wait()
                start_time = ev.profile.START*1e-9
                if not ref[dev]:
                    ref[dev] = start_time
                else:
                    ref[dev] = min(ref[dev],start_time)

            for ev in kernel.read_events:
                ev.wait()
                start_time = ev.profile.START *1e-9
                end_time =   ev.profile.END *1e-9
                if not ref[dev]:
                    ref[dev] = start_time
                else:
                    ref[dev] = min(ref[dev],start_time)



    host_st = None
    host_en = None

    timestamps = {}

    for dag in all_dags:
        print("dag number : ",dag.id)

        for kernel_id,kernel in dag.kernels.items():
            timestamps[kernel.name+str(kernel_id)] = {}
            dev = kernel.task_object.device
            fin = ref[dev]
            kernel_timestamps = timestamps[kernel.name+str(kernel_id)]
            print("\t Kernel ",kernel.name, " ",kernel.id, " ",dev)
            kernel_timestamps["write"] = {"host_queued_start":kernel.host_events[0].write_start,\
            "host_queued_end":kernel.host_events[0].write_end,"device_queued":-1,"device_start":-1,"device_end":-1}
            kernel_timestamps["device"] = dev
            kernel_timestamps["cmdq"] =  kernel.dag_device
            st = None
            for ev in kernel.write_events:
                ev.wait()
                queue_time = ev.profile.QUEUED*1e-9
                start_time = ev.profile.START *1e-9
                end_time =   ev.profile.END *1e-9

                if kernel_timestamps["write"]["device_queued"] == -1:
                    kernel_timestamps["write"]["device_queued"] = queue_time
                else:
                    kernel_timestamps["write"]["device_queued"] = min(queue_time,kernel_timestamps["write"]["device_queued"])

                if kernel_timestamps["write"]["device_start"] == -1:
                    kernel_timestamps["write"]["device_start"] = start_time
                else:
                    kernel_timestamps["write"]["device_start"] = min(start_time,kernel_timestamps["write"]["device_start"])

                if kernel_timestamps["write"]["device_end"] == -1:
                    kernel_timestamps["write"]["device_end"] = end_time
                else:
                    kernel_timestamps["write"]["device_end"] = max(end_time,kernel_timestamps["write"]["device_end"])


                print("\t\t Write event | Start time ",start_time-ref[dev], " | End time ", end_time-ref[dev])
                #kernel_timestamps["write"].append([start_time-ref[dev],end_time-ref[dev]])

                if st == None:
                    st = start_time
                else:
                    st = min(st,start_time)
                fin = max(fin,end_time)

            kernel_timestamps["nd_range"] = {"device_start":-1,"device_end":-1}
#            ev = kernel.nd_range_event
            for ev in kernel.nd_range_event:
                ev.wait()
                start_time = ev.profile.START*1e-9
                end_time =   ev.profile.END*1e-9
                print("\t\t ND range | Start time ",start_time-ref[dev], " | End time ", end_time-ref[dev])
                #kernel_timestamps["nd_range"].append([start_time-ref[dev],end_time-ref[dev]])
                if st==None:
                    st = start_time
                else:
                    st = min(st,start_time)
                fin = max(fin,end_time)

                if kernel_timestamps["nd_range"]["device_start"] == -1:
                    kernel_timestamps["nd_range"]["device_start"] = start_time
                else:
                    kernel_timestamps["nd_range"]["device_start"] = min(start_time,kernel_timestamps["nd_range"]["device_start"])

                if kernel_timestamps["nd_range"]["device_end"] == -1:
                    kernel_timestamps["nd_range"]["device_end"] = end_time
                else:
                    kernel_timestamps["nd_range"]["device_end"] = max(end_time,kernel_timestamps["nd_range"]["device_end"])


            kernel_timestamps["read"] = {"device_start":-1,"device_end":-1}
            for ev in kernel.read_events:
                ev.wait()
                start_time = ev.profile.START*1e-9
                end_time =   ev.profile.END*1e-9
                print("\t\t Read event | Start time ",start_time-ref[dev], " | End time ", end_time-ref[dev])
                #kernel_timestamps["read"].append([start_time-ref[dev],end_time-ref[dev]])
                if st==None:
                    st = start_time
                else:
                    st = min(st,start_time)
                fin = max(fin,end_time)

                if kernel_timestamps["read"]["device_start"] == -1:
                    kernel_timestamps["read"]["device_start"] = start_time
                else:
                    kernel_timestamps["read"]["device_start"] = min(start_time,kernel_timestamps["read"]["device_start"])

                if kernel_timestamps["read"]["device_end"] == -1:
                    kernel_timestamps["read"]["device_end"] = end_time
                else:
                    kernel_timestamps["read"]["device_end"] = max(end_time,kernel_timestamps["read"]["device_end"])



            err = 0

            if args.check_error:
                if kernel.name.endswith("copy"):
                    if kernel.name[:4] == "coal":
                        m = kernel.global_work_size[1]
                        n = kernel.global_work_size[0]
                    else:
                        m = kernel.global_work_size[0]
                        n = kernel.global_work_size[1]
                    inp = kernel.data["input"][0]
                    out = kernel.data["output"][0]
                    inp = inp.reshape(m,n)
                    out = out.reshape(m,n)
                    err = np.mean((out-inp)**2)

                if kernel.name.endswith("transpose"):
                    if kernel.name[:4] == "coal":
                        m = kernel.global_work_size[1]
                        n = kernel.global_work_size[0]
                    else:
                        m = kernel.global_work_size[0]
                        n = kernel.global_work_size[1]
                    inp = kernel.data["input"][0]
                    out = kernel.data["output"][0]
                    inp = inp.reshape(m,n)
                    out = out.reshape(n,m)
                    err = np.mean((out-inp.T)**2)

                if "gemm" in kernel.name.lower():

                    m = kernel.symbolic_variables["m1"]
                    p = kernel.symbolic_variables["p1"]
                    n = kernel.symbolic_variables["n1"]


                    inp1 = kernel.data["input"][0]
                    inp2 = kernel.data["input"][1]
                    bias = kernel.data["input"][2]
                    out = kernel.data["output"][0]
                    inp1 = inp1.reshape(m,p)
                    inp2 = inp2.reshape(p,n)
                    out = out.reshape(m,n)
                    err = np.mean((inp1.dot(inp2)+bias-out)**2)




            #print "\t \t "+str(kernel_timestamps["write"]["host_queued"]-kernel_timestamps["write"]["device_queued"])
            # print "\t \t Time taken(measured by device times stamps)", fin-st

    #         if kernel.host_events[0].read_end and kernel.host_events[0].write_start:
    #             host_total_time = kernel.host_events[0].read_end-kernel.host_events[0].write_start
    #             print "\t \t Time Taken (measured by host time stamps) : ",host_total_time
    #
    #
    #
    #             if host_en == None:
    #                 host_en = kernel.host_events[0].read_end
    #             else:
    #                 host_en = max(host_en,kernel.host_events[0].read_end)
    #
    #             if host_st == None:
    #                 host_st = kernel.host_events[0].write_start
    #             else:
    #                 host_st = min(host_st,kernel.host_events[0].write_start)
    #
    #
    #             total_host_overhead = host_total_time - (fin-st)
    #             print "\t \t Measured Host overhead :",total_host_overhead
    #             print "\t \t Percentage overhead:",total_host_overhead*100/host_total_time
    #             if args.check_error:
    #                 print "\t \t Error :- ",err
    #             print "\n"
    #
    #
    #         else:
    #             print "\t \t Host Profiling data not available, continuing..\n"
    #
    # en = host_en-host_st
    #
    # print "Total Time as measured by Host read callback threads is ",en
    # #print "Total Time as measured by scheduler is ",total_time_in_multiple_dag_devices
    #
    # #print timestamps

    print("\n")
    #print json.dumps(timestamps,sort_keys=True,indent=2)
    #print timestamps
    if args.full_dump_path == "None":
        if args.use_thread:
            with open("./scheduling/dumps/thread.json","w") as f:
                print("saving to thread.json")
                s = json.dumps(timestamps)
                f.write(s)
                # json.dump(timestamps,f)

        else:
            with open("./scheduling/dumps/non_thread.json","w") as f:
                print("saving to non_thread.json")
                s = json.dumps(timestamps)
                f.write(s)
                # json.dump(timestamps,f)


    else:
        with open(args.full_dump_path,"w") as f:
            print("saving to ",args.full_dump_path)
            s = json.dumps(timestamps)
            f.write(s)
            # json.dump(timestamps,f)


    time.sleep(2)
