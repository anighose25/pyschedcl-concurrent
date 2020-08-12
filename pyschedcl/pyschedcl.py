import pyopencl as cl
import pyopencl.tools
import pyopencl.array
import datetime
import collections
import os
from copy import deepcopy
import threading
import mutex
import logging
import numpy as np
import constant_pyschedcl as cons
import time
import gc
import resource
from decimal import *
import threading
import time
from collections import defaultdict


try:
    import Queue as Q
except ImportError:
    import queue as Q

numpy_types = {
    "unsigned": np.uint32,
    "unsigned int": np.uint32,
    "uint": np.uint32,
    "int": np.int32,
    "long": np.int64,
    "long int": np.int64,
    "float": np.float32,
    "double": np.float64,
    "char": np.int8,
    "short": np.int16,
    "uchar": np.uint8,
    "unsigned char": np.uint8,
    "ulong": np.uint64,
    "unsigned long": np.uint64,
    "ushort": np.uint16,
    "unsigned short": np.uint16
}

VEC_TYPES = ['char16', 'char2', 'char3', 'char4', 'char8', 'double16', 'double2', 'double3', 'double4', 'double8',
             'float16', 'float2', 'float3', 'float4', 'float8', 'int16', 'int2', 'int3', 'int4', 'int8', 'long16',
             'long2', 'long3', 'long4', 'long8', 'short16', 'short2', 'short3', 'short4', 'short8', 'uchar16', 'uchar2',
             'uchar3', 'uchar4', 'uchar8', 'uint16', 'uint2', 'uint3', 'uint4', 'uint8', 'ulong16', 'ulong2', 'ulong3',
             'ulong4', 'ulong8', 'ushort16', 'ushort2', 'ushort3', 'ushort4', 'ushort8', ]



SOURCE_DIR = os.getcwd()+"/"
MAX_GPU_ALLOC_SIZE = 0
MAX_CPU_ALLOC_SIZE = 0
# finished_kernels = list()
# q = Q.PriorityQueue()
for datatype in VEC_TYPES:
    numpy_types[datatype] = eval('cl.array.vec.{}'.format(datatype))

system_cpus, system_gpus = 0, 0
nGPU, nCPU = 0, 0
est_cpu = 0.0
est_gpu = 0.0
device_history = {"gpu": [], "cpu": []}
ready_queue = {"gpu": collections.deque(), "cpu": collections.deque()}
cs = mutex.mutex()
user_defined = dict()
dump_output = False
just_for_testing_num_chunks = 1

global_programs = defaultdict(dict)


# bolt = [0]
# rqlock = [0]
boltLock = threading.Lock()
rqLock = threading.Condition()


callback_queue = {}

###################TODO###############################
finishe_ker = list()
ex_callback_queue = {}
ex_callback_queue["READ"] = {}
ex_callback_queue["WRITE"] = {}
ex_callback_queue["KERNEL"] = {}

kernel_hist = {}
TASK = 0
ken = 0
enque_read = 0
enque_write = 0
duplicate_read = 0
duplicate_write = 0
kernel__data = {}
kernel__data1 = {}
kernel_dataset = {}
kernel_chunkleft = {}
kernel_buffer = {}
spantime = 0
release_host = list()
release_device = list()
count = 0
time1 = list()
construction_time = 0
kernel__name = {}
task_dag_object = None
done_events = []


frontier_Q = []
frontier_Q_lock = threading.Condition()

def convert_to_milliseconds(exec_time):
    return 1000*exec_time

def adjust_zero(timestamps):
    kernels = timestamps.keys()
    reference_device = {}
    reference_host = {}
    total_time = 0
    for kernel in kernels:
        device = timestamps[kernel]["device"]
        t = timestamps[kernel]["write"]["device_queued"]
        if t == -1:
            continue
        if not (device in reference_device):
            reference_device[device] = t
        else:
            reference_device[device] = min(reference_device[device],t)

        t = timestamps[kernel]["write"]["host_queued"]
        logging.info("Host Queued time: " + str(t))
        if not (device in reference_host):
            reference_host[device] = t
        else:
            reference_host[device] = min(reference_host[device],t)


    relative_timestamps = deepcopy(timestamps)

    global_reference = None

    for key,value in reference_host.items():
        if not global_reference:
            global_reference = value
        else:
            global_reference = min(value,global_reference)


    for kernel,kernel_timestamps in relative_timestamps.items():
        device = kernel_timestamps["device"]
        for event_type,event_timestamps in kernel_timestamps.items():
            #print(event_type)
            if event_type == "device":
                continue
            else:
                #continue
                for sub_event_type in event_timestamps:
                    if  sub_event_type[:4] == "host":
                        event_timestamps[sub_event_type] -= global_reference
                    else:
                        event_timestamps[sub_event_type] = event_timestamps[sub_event_type] - reference_device[device] + reference_host[device] - global_reference 
                    total_time = max(total_time,event_timestamps[sub_event_type])

    #print "Total Time Taken - ",total_time
    #print(json.dumps(relative_timestamps,sort_keys=True,indent=1))          
    return relative_timestamps, total_time

def replace(dictionary, a, b):
    #print "replacing"
    a = str(a)
    b = str(b)
    #print type(dictionary)

    if type(dictionary) == list:
        for item in dictionary:
            replace(item,a,b)

    elif type(dictionary) == dict:
        for key in dictionary:
            #print dictionary[key],type(dictionary[key])
            if type(dictionary[key]) == dict:
                replace(dictionary[key],a,b)
            elif type(dictionary[key]) in [str,unicode]:
                #print "before replacement : ",dictionary[key]
                dictionary[key] = dictionary[key].replace(a,b)
                #print "after replacement : ",dictionary[key]
            elif type(dictionary[key]) == list:
                for item in dictionary[key]:
                    replace(item,a,b)




def create_dag(info_folder,dag_file,output_file,partition=-1):
    from os import listdir
    from os.path import join
    import json
    dag_info = open(dag_file,'r').readlines()
    counter = 0

    task_map = {}
    task_symvar_map = {}
    while dag_info[counter]!='---\n':
        line = dag_info[counter].strip("\n")
        key,value,symvar = line.split(" ")
        task_map[int(key)] = value
        task_symvar_map[int(key)]=eval(symvar)
        counter +=1
    counter +=1


    adj_list = defaultdict(list)
    buffer_edge_info = []

    while dag_info[counter]!='---\n':
        line = dag_info[counter].strip("\n")
        u,v = line.split("->")
        s,b_s = map(int,u.split(" "))
        d,b_d = map(int,v.split(" "))
        adj_list[d].append((s,b_s,b_d))
        buffer_edge_info.append(line)
        counter +=1
    json_files = [join(info_folder,f) for f in listdir(info_folder)]
    json_dictionary = {}
    dag_json = []

    print adj_list

    for f in listdir(info_folder):
        if f.endswith('.json') and (f in task_map.values()):
            filename = join(info_folder,f)
            with open(filename,'r') as g:
                json_dictionary[f]=json.loads(g.read())

    for t in task_map:
        json_file = deepcopy(json_dictionary[task_map[t]])
        json_file["id"]=t
        json_file["symbolicVariables"]=task_symvar_map[t]
        #print json_file
        for sym,val in json_file["symbolicVariables"].items():
            replace(json_file,sym,val)

        json_file["depends"]=set()
        json_file["partition"] = json_file["symbolicVariables"]["partition"]
        if "localWorkSize" in json_file["symbolicVariables"]:
            json_file["localWorkSize"] = [json_file["symbolicVariables"]["localWorkSize"] for _ in range(json_file["workDimension"])]


        if partition!=-1:
            if int(json_file["partition"]) == 10:
                json_file["task"] = 0
            else:
                json_file["task"] = 1


        # print "Task ",t
        # print adj_list[t]
        for v in adj_list[t]:
            u,s_b,d_b = v
            json_file["depends"].add(u)

            from_value = {"kernel":u,"pos":s_b}
            for buffer_type in ["inputBuffers","ioBuffers"]:
                for buffer_info in json_file[buffer_type]:
                    if buffer_info["pos"]==d_b:
                        buffer_info["from"]=from_value

        json_file["depends"] = list(json_file["depends"])
        dag_json.append(json_file)

    with open(output_file,'w') as g:
        json.dump(dag_json,g,indent=2)



def blank_fn(*args, **kwargs):
    """
    Does nothing. Used as dummy function for callback events.

    """
    pass


class HostEvents(object):
    """
    Class for storing timing information of various events associated with a kernel.

    :ivar dispatch_start: Start Timestamp for dispatch function
    :ivar dispatch_end:  End Timestamp for dispatch function
    :ivar create_buf_start: Start Timestamp for Creation of Buffers
    :ivar create_buf_end: End Timestamp for Creation of Buggers
    :ivar write_start: Start TimeStamp for Enqueuing Write Buffer Commands on Command Queue
    :ivar write_end: End Timestamp for Writing of Buffers to Device
    :ivar ndrange_start: Start TimeStamp for Launching Kernel
    :ivar ndrange_end: End Timestamp for when kernel execution is finished on device
    :ivar read_start:  Start TimeStamp for Enqueuing Read Buffer Commands on Command Queue
    :ivar read_end: End TimeStamp for Reading of Buffers from Device to Host
    :ivar kernel_name: Name of kernel
    :ivar kernel_id: Unique id for kernel
    :ivar dispatch_id: Dispatch id for kernel
    """

    def __init__(self, kernel_name='', kernel_id='', dispatch_id='', dispatch_start=None, dispatch_end=None,
                 create_buf_start=None, create_buf_end=None, write_start=None, write_end=None, ndrange_start=None,
                 ndrange_end=None, read_start=None, read_end=None):
        """
        Initialise attributes of HostEvents class .

        """
        self.dispatch_start = dispatch_start
        self.dispatch_end = dispatch_end
        self.create_buf_start = create_buf_start
        self.create_buf_end = create_buf_end
        self.write_start = write_start
        self.write_end = write_end
        self.ndrange_start = ndrange_start
        self.ndrange_end = ndrange_end
        self.read_start = read_start
        self.read_end = read_end
        self.kernel_name = kernel_name
        self.kernel_id = kernel_id
        self.dispatch_id = dispatch_id

    def __str__(self):

        a = deepcopy(self.__dict__)
        for i in a:
            a[i] = str(a[i])
        return str(a)

    def __repr__(self):

        return str(self)

    def is_not_empty(self):
        empty = self.dispatch_start is None and self.dispatch_end is None and self.create_buf_start is None and self.create_buf_end is None and self.write_start is None and self.write_end is None and self.ndrange_start is None and self.ndrange_end is None and self.read_start is None and self.read_end is None
        return not empty






def dump_device_history():
    """
    Dumps device history to debug.log file.
    """

    debug_strs = []
    min_timestamp = Decimal('Infinity')
    max_timestamp = 0.0
    finishing_timestamps ={'cpu':0.0, 'gpu': 0.0}
    for dev in ['gpu', 'cpu']:
        for device_id in range(len(device_history[dev])):
            for host_event in device_history[dev][device_id]:
                if host_event.is_not_empty():
                    kernel_id = host_event.kernel_id
                    kernel_name = host_event.kernel_name
                    write_start = "%.20f" % host_event.write_start
                    min_timestamp = min(min_timestamp, Decimal(write_start))
                    write_end = "%.20f" % host_event.write_end
                    ndrange_start = "%.20f" % host_event.ndrange_start
                    ndrange_end = "%.20f" % host_event.ndrange_end
                    read_start = "%.20f" % host_event.read_start
                    read_end = "%.20f" % host_event.read_end
                    max_timestamp = max(max_timestamp, Decimal(read_end))
                    debug_str = "HOST_EVENT " + dev + " " + str(device_id) + " " + str(
                        kernel_id) + "," + kernel_name + " " + write_start + " " + write_end + " " + \
                                ndrange_start + " " + ndrange_end + " " + read_start + " " + read_end
                    logging.debug(debug_str)
                    # print debug_str
                    debug_strs.append(debug_str)
                    finishing_timestamps[dev]=round(float(Decimal(read_end)-Decimal(write_start)),4)
    profile_time = max_timestamp - min_timestamp
    print "span_time " + str(profile_time)
    print finishing_timestamps
    # return debug_strs
    return finishing_timestamps


def partition_round(elms, percent, exact=-1, total=100, *args, **kwargs):
    """
    Partitions dataset in a predictable way.

    :param elms: Total Number of elements
    :type elms: Integer
    :param percent: Percentage of problem space to be processed on one device
    :param type: Integer
    :param exact: Flag that states whether percentage of problem space is greater than 50 or not (0 for percent < 50, 1 for percent >= 50)
    :param type: Integer
    :param total: Percentage of total problem space (Default value: 100)
    :type total: Integer
    :return: Number of elements of partitioned dataset
    :rtype: Integer
    """
    if elms < 100:
        factor = 10
        x = elms / 10.0
    else:
        factor = 1
        x = elms / 100.0
    if exact == -1:
        exact = 0 if percent > 50 else 1
    if elms % 2 == 0:
        if percent == 50:
            logging.debug(
                "PARTITION: get_slice_values -> multiple_round -> partition_round (if percent=50) returns: %d",
                elms / 2)
            return int(elms / 2)
        elif exact == 0:
            b = int(x * (total - percent) / factor)
            return partition_round(elms, total) - b if total != 100 else elms - b
        elif exact == 1:
            logging.debug("PARTITION: get_slice_values -> multiple_round -> partition_round (if exact=1) returns: %d",
                          x * percent / factor)
            return int(x * percent / factor)
    else:
        if percent > 50:
            return partition_round(elms - 1, percent, exact, total)
        else:
            return partition_round(elms - 1, percent, exact, total) + 1


part_round = partition_round


def multiple_round(elms, percent, multiples, **kwargs):
    """
    Partitions such that the partitioned datasets are multiples of given number.

    :param elms: Total number of elements of buffer
    :type elms: Integer
    :param percent: Percentage of problem space to be processed by one device
    :type percent: Integer
    :param multiples: List of integers representing partition multiples for each dimension
    :type multiples: list of integers
    :param kwargs:
    :return: Percentage of buffer space to be partitioned for one device
    :rtype: Integer
    """
    if percent == 100.0:
        return elms
    # print "PYSCHEDCL_MULTIPLES", multiples, percent
    for multiple in multiples:
        if elms % multiple == 0 and elms > multiple:
            x = elms / multiple
            return partition_round(x, percent, **kwargs) * multiple


def ctype(dtype):
    """
    Convert a string datatype to corresponding Numpy datatype. User can also define new datatypes using user_defined parameter.

    :param dtype: Datatype name
    :type dtype: String
    :return: Numpy Datatype corresponding to dtype
    """
    global numpy_types
    try:
        return numpy_types[dtype]
    except:
        Exception("Data Type {} not defined".format(dtype))


def make_ctype(dtype):
    """
    Creates a vector datatype.

    :param dtype: Datatype name
    :type dtype: String
    :return: numpy datatype corresponding to dtype
    """
    global numpy_types
    if dtype in VEC_TYPES:
        return eval('cl.array.vec.make_{}'.format(dtype))
    else:
        return numpy_types[dtype]


def make_user_defined_dtype(ctxs, name, definition):

    global numpy_types
    if type(definition) is str:
        if name not in numpy_types:
            if definition not in numpy_types:
                raise Exception(
                    "Cant recognize definition {0} should be one of {1}".format(definition, numpy_types.keys()))
            else:
                numpy_types[name] = numpy_types[definition]
        else:
            if numpy_types[definition] != numpy_types[name]:
                raise Exception(
                    "Conflicting definitions {0} and {1} for {2}".format(numpy_types[definition], numpy_types[name],
                                                                         name))
    elif type(definition) is dict:
        raise NotImplementedError
        struct = np.dtype(map(lambda k, v: (k, numpy_types[v]), definition.items()))
        struct, c_decl = cl.tools.match_dtype_to_c_struct(ctxs['gpu'][0])

    else:
        raise Exception('Expected data type definition to be string or dict but got {}'.format(str(type)))


def notify_callback(kernel, device, dev_no, event_type, events, host_event_info,callback=blank_fn):
    """
    A wrapper function that generates and returns a call-back function based on parameters. This callback function is run whenever a enqueue operation finishes execution. User can suitably modify callback functions to carry out further processing run after completion of enqueue read buffers operation indicating completion of a kernel task.

    :param kernel: Kernel Object
    :type kernel:  pyschedcl.Kernel object
    :param device: Device Type (CPU or GPU)
    :type device: String
    :param dev_no: PySchedCL specific device id
    :type dev_no: Integer
    :param event_type: Event Type (Write, NDRange, Read)
    :type event_type: String
    :param events: List of Events associated with an Operation
    :type events: list of pyopencl.Event objects
    :param host_event_info: HostEvents object associated with Kernel
    :type host_event_info: HostEvents
    :param callback_events: Callback thread events associated with the Kernel to synchronise function callback routines
    :type: Dictionary
    :param callback: Custom Callback function for carrying out further post processing if required.
    :type callback: python function
    """

    def cb(status):
        global boltLock,rqLock
        # boltLock.acquire()
        # rqLock.acquire()
        try:
            global callback_queue
            tid = threading.currentThread()
            callback_queue[tid] = False

            # while (compare_and_swap(0, 1) == 1):
            #     debug_probe_string = "CALLBACK_PROBE : " + kernel.name + " " + str(device) + " " + str(
            #         event_type) + " event"
            #     logging.debug(debug_probe_string)

            lw_callback = kernel.lw_callback

            callback_events = kernel.cb_events
            if not lw_callback:
                relevant_callback_events = callback_events[device][dev_no]

            if (not lw_callback) and event_type != "WRITE":
                relevant_callback_events["write_done"].wait()
                if event_type == "READ":
                    relevant_callback_events["kernel_done"].wait()


            boltLock.acquire()#ensures only one callback function runs at a time


            debug_trigger_string = "CALLBACK_TRIGGERED : " + kernel.name + " " + str(
                event_type) + " execution finished for device " + str(device)


            logging.debug(debug_trigger_string)

            #time.sleep(5) #comment out to test

            if event_type == 'WRITE':
                host_event_info.write_end = time.time()
                if not lw_callback:
                    relevant_callback_events["write_done"].set()

            elif event_type == 'READ':
                host_event_info.read_end = time.time()
                global device_history
                #logging.debug("CALLBACK : " +str(host_event_info))
                #logging.debug("CALLBACK : Pushing info onto " + str(device) + str(dev_no))
                device_history[device][dev_no].append(host_event_info)

                if kernel.multiple_break:
                    if device == 'cpu':
                        kernel.chunks_cpu -= 1
                    else:
                        kernel.chunks_gpu -= 1
                    if device == 'cpu' and kernel.chunks_cpu == 0:
                        kernel.release_buffers(device)

                    if device == 'gpu' and kernel.chunks_gpu == 0:
                        kernel.release_buffers(device)
                else:
                    kernel.release_buffers(device)
                kernel.chunks_left -= 1
                if kernel.chunks_left == 0:
                    global dump_output
                    if dump_output:
                        import pickle
                        filename = SOURCE_DIR + "output/" + kernel.name + "_" + str(kernel.partition) + "_" + str(kernel.dataset) + ".pickle"
                        print "Dumping Pickle"
                        with open(filename, 'wb') as handle:
                            pickle.dump(kernel.data['output'], handle, protocol=pickle.HIGHEST_PROTOCOL)
                        print "Dumped Pickle"
                    #kernel.release_host_arrays()

                global ready_queue
                # while (test_and_set(0, 1)):
                #     pass
                rqLock.acquire()
                ready_queue[device].append(dev_no)
                if device == 'gpu':
                    global nGPU
                    nGPU += 1
                else:
                    global nCPU
                    nCPU += 1
                #global rqlock
                rqLock.release()

            elif event_type == 'KERNEL':
                host_event_info.ndrange_end = time.time()
                if not lw_callback:
                    relevant_callback_events["kernel_done"].set()
            callback_queue[tid] = True
            #bolt[0] = 0
            logging.debug("CALLBACK_RESET : " + kernel.name + " Resetting bolt value by " + str(device) + " " + str(
                event_type) + " event")
            boltLock.release()
        except TypeError:
            pass
        # rqLock.release()
        # boltLock.release()

    return cb

################################TODO#######################################################################
def release_buffers(unfinished_kernel,task , taskdag):
    global release_host , release_device
    unfinished_kernel_copy = unfinished_kernel
    logging.debug("unfinished_kernel", map(lambda x:x.id , unfinished_kernel_copy))
    #print unfinished_kernel
    for i in unfinished_kernel:
        if(i.id in taskdag.kernels.keys()):
            dependents = taskdag.get_kernel_children_ids(int(i.id))
            internal_dependents = set(task.get_kernel_ids()) & set(dependents)
            maps = map(lambda x:x.id , taskdag.unfinished_kernels)
            if(set(maps) & set(internal_dependents) == set(internal_dependents) and i  not in release_device):
                release_device.append(i)
                i.release_times -=1
                if(i.partition == 10):
                    i.release_buffers_io_out_dag('gpu' , taskdag , task)
                else:
                    i.release_buffers_io_out_dag('cpu' , taskdag , task)
            internal_dependents = set(dependents)
            if(set(maps) & set (internal_dependents) == set(internal_dependents) and i not in release_host):
                i.release_host_arrays()
                i.release_times -=1
                release_host.append(i)
            if(i.id not in finishe_ker and i.release_times <= 0):
                finishe_ker.append(i.id)
            if(i in unfinished_kernel and i.release_times == 0):
                unfinished_kernel.remove(i)



def shortest_path_length(G, source=None, target=None, weight=None):
    if source is None:
        if target is None:
            if weight is None:
                paths=nx.all_pairs_shortest_path_length(G)
            else:
                paths=nx.all_pairs_dijkstra_path_length(G, weight=weight)
        else:
            with nx.utils.reversed(G):
                if weight is None:
                    paths=nx.single_source_shortest_path_length(G, target)
                else:
                    paths=nx.single_source_dijkstra_path_length(G, target,
                                                                weight=weight)
    else:
        if target is None:
            if weight is None:
                paths=nx.single_source_shortest_path_length(G,source)
            else:
                paths=nx.single_source_dijkstra_path_length(G,source,weight=weight)
        else:
            if weight is None:
                p=nx.bidirectional_shortest_path(G,source,target)
                paths=len(p)-1
            else:
                paths=nx.dijkstra_path_length(G,source,target,weight)
    return paths

def construct_component(taskdag , task , width, depth):

    global count
    if task is None:
        return

    if not task.is_supertask() and task not in taskdag.component:
        taskdag.component.append(task)

    par = list()
    par1 = list()
    import networkx as n1
    childs = n1.descendants(taskdag.G , task) #list(taskdag.get_task_children(task))
    for succ in childs :
        if(shortest_path_length(taskdag.G, source=task, target=succ, weight=None) <= depth):
            if(succ in list(taskdag.get_tasks_sorted())):
                par = taskdag.get_task_parents(succ)
                for i in list(par):
                    if(shortest_path_length(taskdag.G, source=i, target=succ, weight=None) <= depth):
                        par1.append(i)
            if(succ not in taskdag.component):
                if not succ.is_supertask():
                    add2 = 0
                    for parent in par1 :
                        if(parent != task):
                            if(len(taskdag.get_task_parents(parent)) != 0 ):
                                add2 = 1
                    if(add2 == 0):
                        taskdag.merge_tasks(task , succ)
                    if(succ in taskdag.free_tasks):
                        taskdag.free_tasks_merged.append(succ)
                        taskdag.component.append(succ)

            for parent in par1 :
                if( len(taskdag.get_task_parents(parent)) == 0):
                    if parent not in taskdag.component :
                        if not parent.is_supertask() and parent is not task:
                            if(parent in list(taskdag.get_tasks_sorted())):
                                taskdag.merge_tasks(task , parent)
                                taskdag.component.append(parent)
                                construct_component(taskdag , task, 0 , depth)
                            if(parent in taskdag.free_tasks):
                                taskdag.free_tasks_merged.append(parent)

def rank_calculator(Taskdag):
    for ct in Taskdag.tasks:
        for i in Taskdag.tasks[ct].kernels:
            for j in range(len(i.buffer_info['input'])):
                Taskdag.tasks[ct].rank  =  Taskdag.tasks[ct].rank + i.buffer_info['input'][j]['size']
            for j in range(len(i.buffer_info['io'])):
                Taskdag.tasks[ct].rank  =  Taskdag.tasks[ct].rank + i.buffer_info['io'][j]['size']
            for j in range(len(i.buffer_info['output'])):
                Taskdag.tasks[ct].rank  =  Taskdag.tasks[ct].rank + i.buffer_info['output'][j]['size']



def update_queue(Task_dag):
    global construction_time
    for i in range(len(Task_dag)):
        for j in Task_dag[i].free_tasks:
            if j not in Task_dag[i].component:
                #construct_component(Task_dag[i] , j, 0, 2)
                for j in Task_dag[i].free_tasks_merged:
                    if( j in Task_dag[i].free_tasks):
                        Task_dag[i].free_tasks.remove(j)
        for j in range(len(Task_dag[i].free_tasks)):
            tsk = Task_dag[i].process_free_task()
            q.put(Skill(tsk.rank_b, tsk , Task_dag[i] , tsk.rank_b))

def notify_callback_dag(kernel, device , dev_no, event_type, events, host_event_info,  callback=blank_fn ):
    """
    A wrapper function that generates and returns a call-back function based on parameters. This callback function is run whenever a enqueue operation finishes execution. User can suitably modify callback functions to carry out further processing run after completion of enqueue read buffers operation indicating completion of a kernel task.

    :param kernel: Kernel Object
    :type kernel:  pyschedcl.Kernel object
    :param device: Device Type (CPU or GPU)
    :type device: String
    :param dev_no: PySchedCL specific device id
    :type dev_no: Integer
    :param event_type: Event Type (Write, NDRange, Read)
    :type event_type: String
    :param events: List of Events associated with an Operation
    :type events: list of pyopencl.Event objects
    :param host_event_info: HostEvents object associated witq.put(Skill(tsk.rank_b, tsk , Task_dag[i] , tsk.rank_b))Kernel
    :type host_event_info: HostEvents
    :param callback: Custom Callback function for carrying oq.put(Skill(tsk.rank_b, tsk , Task_dag[i] , tsk.rank_b)) further post processing if required.
    :type callback: python function
    """
    global kernel__data , kernel__data1
    def cb(status):
        global bolt
        global check
        if event_type == 'READ':
            host_event_info.read_end = time.time()
        try:
            # print "printing status of ndrange barrier inside notifycallback",kernel.nd_range_barrier.get_info(cl.event_info.COMMAND_EXECUTION_STATUS) #CRITICAL: To comment or not to comment?
            old =time.time()
            global callback_queue
            tid = threading.currentThread()
            callback_queue[tid] = False

            # while (compare_and_swap(0, 1) == 1):
            #     debug_probe_string = "CALLBACK_PROBE : " + kernel.name + " " + str(device) + " " + str(
            #         event_type) + " event"
            #     logging.debug(debug_probe_string)
            #boltLock.acquire()


            debug_trigger_string = "CALLBACK_TRIGGERED : " + str(kernel.id) + " " + str(
                event_type) + " execution finished for device " + str(device)
            logging.debug(debug_trigger_string)

            if event_type == 'WRITE':
                host_event_info.write_end = time.time()
            elif event_type == 'READ':
                #host_event_info.read_end = time.time()
                kernel__data1[kernel.id] = time.time()
                global device_history
                logging.debug("CALLBACK : " +str(host_event_info))
                logging.debug("CALLBACK : Pushing info onto " + str(device) + str(dev_no))
                device_history[device][dev_no].append(host_event_info)

                if kernel.multiple_break:
                    if device == 'cpu':
                        kernel.chunks_cpu -= 1
                    else:
                        kernel.chunks_gpu -= 1
                kernel.chunks_left -= 1

                logging.info("Chunks Left: "+ str(kernel.chunks_left))


                if kernel.chunks_left == 0:
                    
                    #kernel__data[kernel.id] = time.time() -kernel__data[kernel.id]
                    kernel_chunkleft[kernel.id] = time.time()
                    #Step 1 - add kernel to finished kernels list of task object
                    task = kernel.task_object
                    dag = task.task_dag_object


                    info_trigger_string = "EVENT: " + str(kernel.id) + " finished on device " + str(device)
                    logging.info(info_trigger_string)
                    #Step 2 - update succesors
                    succesors = dag.get_kernel_children_ids(kernel.id)
                    logging.info("Successors being added" + str(succesors))
                    child_tasks_to_be_dispatched = []
                    for child in succesors:

                        child_kernel = dag.kernels[child]
                        child_task = child_kernel.task_object



                        if child_task.id != task.id:
                            dag.kernel_flags[child].finished_parents +=1


                            if dag.kernel_flags[child].is_ready():
                                child_task.resource_lock.acquire()
                                child_task.add_free_kernels([child_kernel])
                                child_task.resource_lock.notifyAll()
                                child_task.resource_lock.release()
                                #Case 1 - child is part of a task that has
                                # already dispatched - this is enough

                                #Case 2 - > else new task needs to be added to the FQ
                                frontier_Q_lock.acquire()
                                if not child_task.has_enqueued:
                                    logging.info("FQ: Adding to frontier queue task containing kernel " + str(child))
                                    frontier_Q.append(child_task)
                                    child_task.has_enqueued = True
                                else:
                                    logging.info("TDQ: Adding to child_tasks_to_be_dispatched task containing kernel" + str(child))
                                    child_tasks_to_be_dispatched.append(child_task)
    #                                child_task.dispatch_all()
                                frontier_Q_lock.notifyAll()
                                frontier_Q_lock.release()





                    #Step 3 :- check if task has finished




                    for task in child_tasks_to_be_dispatched:

                        task.dispatch_all()






            elif event_type == 'KERNEL':
                host_event_info.ndrange_end = time.time()

            callback_queue[tid] = True
            #bolt[0] = 0

            logging.debug("CALLBACK_RESET : " + str(kernel.id) + " Resetting bolt value by " + str(device) + " " + str(
                event_type) + " event")
            extime = time.time()  -  old
            global ex_callback_queue
            ex_callback_queue[event_type][tid] = extime

            #boltLock.release()


        except TypeError:
            pass

    return cb
#########################################################################################################################
def generate_unique_id():
    """
    Generates and returns a unique id string.

    :return: Unique ID
    :rtype: String
    """
    import uuid
    return str(uuid.uuid1())


class Kernel(object):
    """
    Class to handle all operations performed on OpenCL kernel.

    :ivar dataset: An integer representing size of the data on which kernel will be dispatched.
    :ivar id: An id that is used identify a kernel uniquely.
    :ivar eco: A dictionary mapping between size of dataset and Estimated Computation Overhead
    :ivar name: Name of the Kernel
    :ivar src: Path to the Kernel source file.
    :ivar partition: An integer denoting the partition class of the kernel.
    :ivar work_dimension: Work Dimension of the Kernel.
    :ivar global_work_size: A list denoting global work dimensions along different axes.
    :ivar local_work_size: A list denoting local work dimensions along different axes.
    :ivar buffer_info: Properties of Buffers
    :ivar input_buffers: Dictionaries containing actual cl.Buffer objects.
    :ivar output_buffers: Dictionaries containing actual cl.Buffer objects.
    :ivar io_buffers: Dictionaries containing actual cl.Buffer objects.
    :ivar data: Numpy Arrays maintaining the input and output data of the kernels.
    :ivar buffer_deps: Dictionary mapping containing buffer dependencies.
    :ivar variable_args: Data corresponding to Variable arguments of the kernel.
    :ivar local_args: Information regarding Local Arguments of the kernel.
    :ivar kernel objects: Dictionary mapping between devices and compiled and built pyopencl.Kernel objects.
    :ivar events: Dictionary containing pyschedcl.KEvents.
    :ivar source: String containing contents of kernel file.
    :ivar clevents: Dictionary containing pyopencl.Events.
    """

    def __init__(self, src, task_dag_object=None, dataset=1024, partition=None, identifier=None):
        """
        Initialise attributes of Kernel event.

        """
        self.rank = 0
        self.task_object = None
        self.exec_time={'cpu':0.0,'gpu':0.0}
        self.dataset = dataset
        self.symbolic_variables = src["symbolicVariables"]
        
        if 'id' in src:
            #print "Source ID", src['id']
            self.id = int(src['id'])
        else:
            #print "Random ID"
            self.id = generate_unique_id()
        self.optm_device = 0
        #if identifier is not None:
         #   print "Identifier ID"
          #  self.id = identifieri
        ################ TODO: Remove global version of task dag object##############################
        if task_dag_object != None:
            self.task_dag_object = task_dag_object
        else:
            self.task_dag_object = 0
        self.release_times = 2
        ###############################################
        if 'ecos' in src and str(dataset) in src['ecos']:
            self.eco = src['ecos'][str(dataset)]
        elif 'eco' in src:
            self.eco = src['eco']
        else:
            self.eco = 1
        self.name = src['name']
        self.src = src['src']
        self.src_cpu = src['src']
        self.src_gpu = src['src']
        #self.partition = src['partition']
        if partition is not None:
            self.partition = partition
        else:
            self.partition = src['partition']
        self.work_dimension = src['workDimension']
        self.global_work_size = src['globalWorkSize']
        
        if type(self.global_work_size) in [str, unicode]:
            self.global_work_size = eval(self.global_work_size)
        if type(self.global_work_size) is int:
            self.global_work_size = [self.global_work_size]


        if 'localWorkSize' in src:
            self.local_work_size = src['localWorkSize']
        else:
            self.local_work_size = []
        if 'localChunkFactor' in src:
            self.local_chunk=src['localChunkFactor']
        else:
            self.local_chunk=[]
        self.global_work_offset = []
        if 'globalChunkFactor' in src:
            self.global_chunk=src['localChunkFactor']
        else:
            self.global_chunk=[]

        if type(self.local_work_size) in [str, unicode]:
            self.local_work_size = eval(self.local_work_size)
        elif type(self.local_work_size) is int:
            self.local_work_size = [self.local_work_size]
        self.buffer_info = dict()
        self.macros = dict()
        if 'macros_values' in src:
            self.macros=src['macros_values']
        if 'inputBuffers' in src:
            self.buffer_info['input'] = src['inputBuffers']
        else:
            self.buffer_info['input'] = []
        if 'outputBuffers' in src:
            self.buffer_info['output'] = src['outputBuffers']
        else:
            self.buffer_info['output'] = []
        if 'ioBuffers' in src:
            self.buffer_info['io'] = src['ioBuffers']
        else:
            self.buffer_info['io'] = []

        ##changed to support chunking
        self.input_buffers = {'gpu': dict(), 'cpu': dict()}
        self.output_buffers = {'gpu': dict(), 'cpu': dict()}
        self.io_buffers = {'gpu': dict(), 'cpu': dict()}
        self.data = {}
        self.buffer_deps = {}


        if 'varArguments' in src:
            self.variable_args = deepcopy(src['varArguments'])
            self.vargs = src['varArguments']
        else:
            self.variable_args = []
            self.vargs = []
        if 'cpuArguments' in src:
            self.cpu_args = src['cpuArguments']
            print "Ignoring CPU Arguments"
        if 'gpuArguments' in src:
            self.gpu_args = src['gpuArguments']
            print "Ignoring GPU Arguments"
        if 'localArguments' in src:
            self.local_args = src['localArguments']
            for i in range(len(self.local_args)):
                self.local_args[i]['size'] = eval(self.local_args[i]['size'])
        else:
            self.local_args = []
            # self.buffer_info['local'] = deepcopy(self.local_args)
        self.kernel_objects = dict()
        for btype in ['input', 'output', 'io']:
            for i in range(len(self.buffer_info[btype])):
                if type(self.buffer_info[btype][i]['size']) in [str, unicode]:
                    self.buffer_info[btype][i]['size'] = eval(self.buffer_info[btype][i]['size'])
                if 'chunk' in self.buffer_info[btype][i] and type(self.buffer_info[btype][i]['chunk']) in [str,
                                                                                                           unicode]:
                    self.buffer_info[btype][i]['chunk'] = eval(self.buffer_info[btype][i]['chunk'])
                self.buffer_info[btype][i]['create'] = True
                self.buffer_info[btype][i]['enq_write'] = True
                self.buffer_info[btype][i]['enq_read'] = True
                if 'from' in self.buffer_info[btype][i]:
                    self.buffer_deps[self.buffer_info[btype][i]['pos']] = (self.buffer_info[btype][i]['from']['kernel'],
                                                                           self.buffer_info[btype][i]['from']['pos'])

        self.partition_multiples = self.get_partition_multiples()
        self.events = {'gpu': dict(), 'cpu': dict()}
        self.source = None
        # self.clevents = {'gpu': dict(), 'cpu': dict()}
        self.chunks_left = 1
        self.multiple_break = False
        self.chunks_cpu = 0
        self.chunks_gpu = 0

        self.write_events = []
        self.read_events = []
        self.nd_range_event = []




    def get_device_requirement(self):

        req = {'gpu': 0, 'cpu': 0, 'all': 0}
        if self.partition > 0:
            req['gpu'] += 1
            req['all'] += 1
        if self.partition < 10:
            req['cpu'] += 1
            req['all'] += 1
        return req

    def dump_json(self):
        import json
        dump_js = dict()
        dump_js['src'] = self.src
        dump_js['dataset'] = self.dataset
        dump_js['id'] = self.id
        dump_js['name'] = self.name
        dump_js['ecos'] = dict()
        dump_js['ecos'][str(self.dataset)]    = self.eco
        dump_js['macros_values']  =  self.macros
        dump_js['partition']  = self.partition
        dump_js['workDimension'] = self.work_dimension
        dump_js['globalWorkSize'] = self.global_work_size
        dump_js['localWorkSize'] =  self.local_work_size
        dump_js['inputBuffers'] = deepcopy(self.buffer_info['input'])
        for i in dump_js['inputBuffers']:
            i['size'] = str(i['size'])
        dump_js['outputBuffers'] =  deepcopy(self.buffer_info['output'])
        for i in dump_js['outputBuffers']:
            i['size'] = str(i['size'])
        dump_js['ioBuffers'] =   self.buffer_info['io']
        if hasattr(self, 'variable_args'):
            dump_js['varArguments']  = self.variable_args
        if hasattr(self, 'vargs'):
            dump_js['varArguments'] = self.vargs
        if hasattr(self, 'cpu_args'):
            dump_js['cpuArguments'] = deepcopy(self.cpu_args)
            for i in dump_js['cpuArguments']:
                i['size'] = str(i['size'])
        if hasattr(self, 'gpu_args'):
            dump_js['gpuArguments'] = deepcopy(self.gpu_args)
            for i in dump_js['gpuArguments']:
                i['size'] = str(i['size'])
        if hasattr(self, 'local_args'):
            dump_js['localArguments'] = deepcopy(self.local_args)
            for i in dump_js['localArguments']:
                i['size'] = str(i['size'])

        return deepcopy(dump_js)


    def get_num_global_work_items(self):
        """
        Returns the total number of global work items based on global work size.

        :return: Total number of global work items considering all dimensions
        :rtype: Integer
        """
        i = 1
        for j in self.global_work_size:
            i *= j
        return i

    # TODO: Modify to handle dependent buffers.

    def release_host_arrays(self):
        """
        Forcefully releases all host array data after completion of a kernel task
        """
        # for array_type in self.data.keys():
        #     for array in self.data[array_type]:
        #         del array
        # logging.debug("Releasing host arrays")
        del self.data
        gc.collect()

    def release_buffers(self, obj):
        """
        Releases all buffers of a Kernel Object for a particular device given in obj

        :param obj: Specifies Kernel object
        :type obj: String
        """
        debug_str = "Releasing buffers of " + self.name + " on " + obj
        logging.debug(debug_str)
        for i, buff in self.input_buffers[obj].iteritems():
            if buff is not None:
                buff.release()
        for i, buff in self.output_buffers[obj].iteritems():
            if buff is not None:
                buff.release()

        for i, buff in self.io_buffers[obj].iteritems():
            if buff is not None:
                buff.release()

    def release_buffers_dag(self, obj):
        """
        Releases all buffers of a Kernel Object for a particular device given in obj

        :param obj: Specifies Kernel object
        :type obj: String
        """
        debug_str = "Releasing buffers of " + self.name + " on " + obj
        logging.debug(debug_str)
        for i, buff in self.input_buffers[obj].iteritems():
            if buff is not None:
                buff.release()

    def release_buffers_io_out_dag(self, obj, taskdag , task):

        debug_str = "Releasing buffers of " + self.name + " on " + obj
        logging.debug(debug_str)
        if( len(taskdag.get_kernel_children_ids(int(self.id))) == 0 ):

            for i, buff in self.output_buffers[obj].iteritems():
                if buff is not None:
                    buff.release()

            for i, buff in self.io_buffers[obj].iteritems():
                if buff is not None:
                    buff.release()
        else:
            pos = []
            par = taskdag.get_kernel_children_ids(int(self.id))
            par1 = []
            for i in list(par):
                if( i in taskdag.skeleton.nodes()):
                        par1.append(i)
            child_ids = set(taskdag.tasks[self.id].get_kernel_ids()) & set(par1)
            for i in list(child_ids):
                kernel = taskdag.get_kernel(i)
                for j in kernel.buffer_info['input']:
                    for j1 in j.keys():
                        if( j1 == 'from'):
                            if(int(j['from']['kernel']) == int(self.id)):
                                pos1 = list()
                                pos1.append(j['from']['pos'])
                                pos.append(pos1)
                                del pos1

            for i, buff in self.output_buffers[obj].iteritems():
                match = 1
                for j in pos:
                    if j  != i :
                        match = 0
                if(match):
                    if buff is not None:
                        buff.release()

    def eval_vargs(self, partition=None, size_percent=0, offset_percent=0, reverse=False, exact=-1, total=100):
        """
        Method to evaluate kernel arguments. Evaluates variable kernel arguments from the specification file if they are an expression against size percent.

        :param partition: Partition Class Value
        :type partition: Integer
        :param size_percent: Percentage of problem space to be processed on device
        :type size_percent: Integer
        :param offset_percent: Offset Percentage required
        :type offset_percent: Integer
        :param reverse: Flag for inverting size and offset calculations for respective devices
        :type reverse: Boolean
        :param exact: Flag that states whether percentage of problem space is greater than 50 or not (0 for percent < 50, 1 for percent >= 50)
        :param type: Integer
        :param total: Percentage of total problem space (Default value: 100)
        :type total: Integer
        """

        def partition_round(elms, percent, exact=exact, total=total):
            return part_round(elms, percent, exact, total)

        if partition is not None:
            size_percent = partition * 10
            offset_percent = 0
            if reverse:
                offset_percent = partition * 10
                partition = 10 - partition
                size_percent = partition * 10

        sym = False

        dataset = self.dataset



        if self.vargs:
            for i in range(len(self.vargs)):
                if type(self.vargs[i]['value']) in [str, unicode]:
                    self.variable_args[i]['value'] = eval(self.vargs[i]['value'])


    def get_partition_multiples(self):
        """
        Determines partition multiples based on work dimension. This method returns a list of numbers based on global work size and local work size according to which the partition sizes will be determined.

        :return: List of integers representing partition multiples for each dimension

        """
        multiples = [1]
        if self.work_dimension == 1:
            if not self.local_work_size:
                multiples = [1]
            else:
                multiples = [self.local_work_size[0], 1]
            if self.local_chunk:
                multiples[0]=self.local_chunk[0]*multiples[0]

        elif self.work_dimension == 2:
            if not self.local_work_size:
                multiples = [self.global_work_size[1], 1]
            else:
                multiples = [self.local_work_size[0] * self.global_work_size[1], self.global_work_size[1],
                             self.local_work_size[0], 1]
        elif self.work_dimension == 3:
            if not self.local_work_size:
                multiples = [self.global_work_size[1] * self.global_work_size[2], self.global_work_size[1], 1]
        else:
            print("Invalid Work Dimension")
        return multiples

    def build_kernel(self, gpus, cpus, ctxs,profile=True):
        """
        Builds Kernels from the directory kernel_src/ for each device and stores binaries in self.kernel_objects dict.

        :param gpus: List of OpenCL GPU Devices
        :type gpus: list of pyopencl.device objects
        :param cpus: List of OpenCL GPU Devices
        :type cpus: list of pyopencl.device objects
        :param ctxs: Dictionary of contexts keyed by device types
        :type ctxs: dict
        :return: Dictionary of key: value pairs where key is device type and value is the device specific binary compiled for the kernel
        :rtype: dict
        """
        global global_programs
        start = time.time()
        if profile:
            src_path = SOURCE_DIR + '/database/kernels/' + self.src
            src_path_cpu = SOURCE_DIR + '/database/kernels/' + self.src_cpu
            src_path_gpu = SOURCE_DIR + '/database/kernels/' + self.src_gpu

        else:
            src_path = SOURCE_DIR + '/kernel_src/' + self.src
            src_path_cpu = SOURCE_DIR + '/kernel_src/' + self.src_cpu
            src_path_gpu = SOURCE_DIR + '/kernel_src/' + self.src_gpu


        #print src_path,"printing source path"

        if not os.path.exists(src_path):
            raise IOError('Kernel Source File %s not Found' % src_path)
        if not os.path.exists(src_path_cpu):
            raise IOError('Kernel Source File for CPU %s not Found' % src_path)
        if not os.path.exists(src_path_gpu):
            raise IOError('Kernel Source File for GPU %s not Found' % src_path)


        if src_path in global_programs:
            self.kernel_objects["gpu"] = global_programs[src_path]["gpu"]
            self.kernel_objects["cpu"] = global_programs[src_path]["cpu"]
            logging.info("Program Already Built... Returning... time taken - "+str(time.time()-start))
            return

        with open(src_path,"r") as f:
            self.source = f.read()
        self.source_gpu = self.source
        self.source_cpu = self.source
        logging.info("Source read time - "+str(time.time()-start))

        # self.source_cpu = open(src_path_cpu).read()
        # self.source_gpu = open(src_path_gpu).read()
        programs = dict()
        opt = ""
        for key in self.macros.keys():
            opt+=(" -D "+key+"="+str(self.macros[key]))
        # opt+=" -cl-single-precision-constant"
        opt+=" -cl-mad-enable"

        start = time.time()
        for key in ctxs.keys():
            if ctxs[key] is not None:
                if key == "cpu":
                    programs[key] = cl.Program(ctxs[key], self.source_cpu)
                if key == "gpu":
                    programs[key] = cl.Program(ctxs[key], self.source_gpu)
        logging.info("Initialization time - "+str(time.time()-start))



        start = time.time()
        if len(gpus) != 0:
            programs['gpu'].build(options=opt,devices=gpus)
        if len(cpus) != 0:
            programs['cpu'].build(options=opt,devices=cpus)

        logging.info("Build  time - "+str(time.time()-start))
        for key in programs.keys():
            self.kernel_objects[key] = cl.Kernel(programs[key], self.name)
            global_programs[src_path][key] = self.kernel_objects[key]



        return programs


    def get_data_types_and_shapes(self):
        for btype in ['input', 'io','output']:
            if btype in self.data.keys():
                for data_buf in self.data[btype]:
                    print btype, type(data_buf), data_buf.shape, data_buf.dtype
        for local_data_buf in self.local_args:
            print "local", local_data_buf['size']

    def random_data(self, low=0, hi=4096):
        """
        Generates random data numpy arrays according to buffer type so that it can be enqueued to buffer. Can be used for testing. Will not generate random data for those buffers that are already enqueued. Creates empty arrays for read-only buffers. Populates values for self.data dictionary.

        :param low: Lower value of data array size
        :type low: Integer
        :param hi: Higher value of data array size
        :type hi: Integer

        """
        import numpy as np
        integers = ['int', 'uint', 'unsigned', 'long', 'unsigned int', 'long int', 'int16', 'int2', 'int3', 'int4',
                    'int8', 'long16', 'long2', 'long3', 'long4', 'long8', 'short16', 'short2', 'short3', 'short4',
                    'short8', 'uint16', 'uint2', 'uint3', 'uint4', 'uint8', 'ulong16', 'ulong2', 'ulong3',
                    'ulong4', 'ulong8', 'ushort16', 'ushort2', 'ushort3', 'ushort4', 'ushort8']
        characters = ['char16', 'char2', 'char3', 'char4', 'char8', 'uchar16', 'uchar2',
                      'uchar3', 'uchar4', 'uchar8']
        for btype in ['input', 'io']:
            self.data[btype] = []
            for i in range(len(self.buffer_info[btype])):
                if not self.buffer_info[btype][i]['enq_write']:
                    self.data[btype].append(None)
                elif self.buffer_info[btype][i]['type'] in integers:
                    self.data[btype].append(
                        np.random.randint(low, hi, size=[self.buffer_info[btype][i]['size']]).astype(
                            ctype(self.buffer_info[btype][i]['type']), order='C'))
                elif self.buffer_info[btype][i]['type'] in characters:
                    self.data[btype].append(
                        np.random.randint(low, 128, size=[self.buffer_info[btype][i]['size']]).astype(
                            ctype(self.buffer_info[btype][i]['type']), order='C'))
                else:
                    self.data[btype].append(np.random.rand(self.buffer_info[btype][i]['size']).astype(
                        ctype(self.buffer_info[btype][i]['type']), order='C'))
        self.data['output'] = []
        for i in range(len(self.buffer_info['output'])):
            self.data['output'].append(
                np.zeros(self.buffer_info['output'][i]['size'], dtype=ctype(self.buffer_info['output'][i]['type']),
                         order='C'))

    def load_data(self, data):
        """
        Populates all host input arrays with given data.

        :param data: Dictionary structure comprising user defined data for host input arrays
        :type data: dict
        """
        import weakref
        import numpy as np
        # print "PYSCHEDCL LOADING", data
        for key in data.keys():
            self.data[key] = []

            for i in range(len(self.buffer_info[key])):
                # if "weakproxy" not in str(type(data[key][i])):
                #     t = weakref.proxy(data[key][i])
                #     self.data[key].append(t)
                # else:
                #     self.data[key].append(data[key][i])
                self.data[key].append(data[key][i])
        self.data['output'] = []
        for i in range(len(self.buffer_info['output'])):
            self.data['output'].append(
                np.zeros(self.buffer_info['output'][i]['size'], dtype=ctype(self.buffer_info['output'][i]['type']),
                         order='C'))
        # print "PYSCHEDCL LOADED", self.data
    def get_data(self, pos):
        """
        Returns the data of a particular kernel argument given its parameter position in the kernel.  Used to load dependent data.

        :param pos: Position of buffer argument in list of kernel arguments
        :type pos: Integer
        :return: Data stored in buffer specified by parameter position in kernel
        :rtype: Numpy array
        """
        for key in self.buffer_info.keys():
            for i in range(len(self.buffer_info[key])):
                if self.buffer_info[key][i]['pos'] == pos:
                    if key in self.data.keys():
                        return self.data[key][i]
                    else:
                        raise KeyError

    def get_buffer_info_location(self, pos):
        """
        Returns buffer_info location at given position. Used to make reusable buffers.

        :param pos: Position of buffer argument in list of kernel arguments
        :type pos: Integer
        :return: Tuple(key, i) where key represents type of buffer access and i represents the id for that buffer in self.buffer_info[key]
        :rtype: Tuple
        """
        for key in self.buffer_info.keys():
            for i in range(len(self.buffer_info[key])):
                if self.buffer_info[key][i]['pos'] == pos:
                    return key, i

    def get_buffer_info(self, pos):
        """
        Returns buffer information stored in Kernel specification file for buffer at given position in Kernel Arguments. Used to make reusable buffers.

        :param pos: Position of buffer argument in list of kernel arguments
        :type pos: Integer
        :return: Returns a dictionary of key:value pairs representing information of selected buffer in self.buffer_info
        :rtype: dict
        """
        key, i = self.get_buffer_info_location(pos)
        return self.buffer_info[key][i]

    def get_buffer(self, pos):
        """
        Returns cl.Buffer objects given its parameter position in the kernel.

        :param pos: Position of buffer argument in list of kernel arguments
        :type: Integer
        :return: PyOpenCL Buffer Object for selected buffer
        :rtype: pyopencl.Buffer
        """
        btype, i = self.get_buffer_info_location(pos)
        if btype is 'input':
            return {'gpu': self.input_buffers['gpu'].get(i, None), 'cpu': self.input_buffers['cpu'].get(i, None)}
        elif btype is 'io':
            return {'gpu': self.io_buffers['gpu'].get(i, None), 'cpu': self.io_buffers['cpu'].get(i, None)}
        elif btype is 'output':
            return {'gpu': self.output_buffers['gpu'].get(i, None), 'cpu': self.output_buffers['cpu'].get(i, None)}
        else:
            raise Exception('Expected buffer to be either input, io or output but got ' + str(btype))

    def check_pow_2(self,n):
        return (n & (n-1) == 0) and n != 0

    def get_chunking_indices(self,size,chunk_number,n_chunks):
        #size = buffer_info['size']


        assert size % n_chunks == 0, "buffer/ndrange cant be divided into equal chunks"
        chunk_size = size//n_chunks


        return chunk_number*chunk_size,chunk_size



    def get_slice_values(self, buffer_info, size_percent, offset_percent, **kwargs):
        """
        Returns Element offset, size based on size_percent, offset_percent.

        :param buffer_info: Dictionary of key:value pairs representing information of one buffer
        :type buffer_info: dict
        :param size_percent: Size of buffer to be processed (in percentage)
        :type size_percent: Float
        :param offset_percent: Offset for given buffer (in percentage)
        :type offset_percent: Float
        :param kwargs:
        :type kwargs:
        :return: Tuple representing element offset and number of elements
        :rtype: Tuple

        """
        logging.debug("PARTITION : %s get_slice_values -> Original Buffer Size: %s", self.name, buffer_info['size'])
        if 'chunk' in buffer_info:
            partition_multiples = [buffer_info['chunk']] + self.partition_multiples
        else:
            partition_multiples = self.partition_multiples

        if buffer_info['break'] != 1:
            eo = 0
            ne = buffer_info['size']
        else:

            if 'exact' not in kwargs:
                eo = multiple_round(buffer_info['size'], offset_percent, partition_multiples, **kwargs)
                ne = multiple_round(buffer_info['size'], size_percent, partition_multiples, **kwargs)
            else:
                eo = multiple_round(buffer_info['size'], offset_percent, partition_multiples, exact=1)
                ne = multiple_round(buffer_info['size'], size_percent, partition_multiples, **kwargs)
        return eo, ne

    def create_buffers(self, ctx, dev, size_percent=100, offset_percent=0, **kwargs):
        """
        Creates buffers for a Context while respecting partition information provided for every buffer associated with the kernel.

        :param ctx: PyOpenCL Context for either a CPU or a GPU device
        :type ctx: pyopencl.Context
        :param dev: Device Type
        :type dev: String
        :param size_percent: Size of valid buffer space for device
        :type size_percent: Float
        :param offset_percent: Offset parameter representing
        :type offset_percent:
        :param kwargs:
        :type kwargs:
        """

        logging.debug("PARTITION : Creating Input Buffers %s",dev)
        for i in range(len(self.buffer_info['input'])):
            if self.buffer_info['input'][i]['create']:
                eo, ne = self.get_slice_values(self.buffer_info['input'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s : %s Create_Input_Buffers_element_offset : %s", dev, self.name, eo)
                logging.debug("PARTITION_%s : %s Create_Input_Buffers_number_of_elements : %s", dev, self.name, ne)
                self.input_buffers[dev][i] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                                                       size=self.data['input'][i][eo:eo + ne].nbytes)

        logging.debug("PARTITION : Creating Output Buffers %s", dev)
        for i in range(len(self.buffer_info['output'])):
            if self.buffer_info['output'][i]['create']:
                eo, ne = self.get_slice_values(self.buffer_info['output'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s: %s Create_Output_Buffers_element_offset : %s", dev, self.name, eo)
                logging.debug("PARTITION_%s: %s Create_Output_Buffers_number_of_elements : %s", dev, self.name, ne)
                self.output_buffers[dev][i] = cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                                                        size=self.data['output'][i][eo:eo + ne].nbytes)

        logging.debug("PARTITION : Creating IO Buffers %s",dev)
        for i in range(len(self.buffer_info['io'])):
            if self.buffer_info['io'][i]['create']:
                eo, ne = self.get_slice_values(self.buffer_info['io'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s: %s Create_IO_Buffers_element_offset : %s", dev, self.name, eo)
                logging.debug("PARTITION_%s: %s Create_IO_Buffers_number_of_elements : %s", dev, self.name, ne)
                self.io_buffers[dev][i]= cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                                                    size=self.data['io'][i][eo:eo + ne].nbytes)

    def create_buffers2(self, ctx, dev, size_percent=100, offset_percent=0, **kwargs):
        """
        Creates buffers for a Context while respecting partition information provided for every buffer associated with the kernel.

        :param ctx: PyOpenCL Context for either a CPU or a GPU device
        :type ctx: pyopencl.Context
        :param dev: Device Type
        :type dev: String
        :param size_percent: Size of valid buffer space for device
        :type size_percent: Float
        :param offset_percent: Offset parameter representing
        :type offset_percent:
        :param kwargs:
        :type kwargs:
        """


        logging.debug("PARTITION : Creating Input Buffers %s",dev)
        for i in range(len(self.buffer_info['input'])):
            if self.buffer_info['input'][i]['create']:
                eo, ne = self.get_slice_values(self.buffer_info['input'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s : %s Create_Input_Buffers_element_offset : %s", dev, self.name, eo)
                logging.debug("PARTITION_%s : %s Create_Input_Buffers_number_of_elements : %s", dev, self.name, ne)
                if dev == 'gpu':
                    self.input_buffers[dev][i] = cl.Buffer(ctx, cl.mem_flags.READ_ONLY,
                                                     size=self.data['input'][i][eo:eo + ne].nbytes)
                else:
                    self.input_buffers[dev][i] = cl.Buffer(ctx, cl.mem_flags.READ_ONLY|cl.mem_flags.USE_HOST_PTR,
                                                     hostbuf=self.data['input'][i][eo:eo+ne])


        logging.debug("PARTITION : Creating Output Buffers %s", dev)
        for i in range(len(self.buffer_info['output'])):
            if self.buffer_info['output'][i]['create']:
                eo, ne = self.get_slice_values(self.buffer_info['output'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s: %s Create_Output_Buffers_element_offset : %s", dev, self.name, eo)
                logging.debug("PARTITION_%s: %s Create_Output_Buffers_number_of_elements : %s", dev, self.name, ne)
                if dev == 'gpu':
                    self.output_buffers[dev][i] = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY,
                                                        size=self.data['output'][i][eo:eo + ne].nbytes)
                else:
                    self.output_buffers[dev][i] = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY|cl.mem_flags.USE_HOST_PTR,
                                                        hostbuf=self.data['output'][i][eo:eo+ne])

        logging.debug("PARTITION : Creating IO Buffers %s",dev)
        for i in range(len(self.buffer_info['io'])):
            if self.buffer_info['io'][i]['create']:
                eo, ne = self.get_slice_values(self.buffer_info['io'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s: %s Create_IO_Buffers_element_offset : %s", dev, self.name, eo)
                logging.debug("PARTITION_%s: %s Create_IO_Buffers_number_of_elements : %s", dev, self.name, ne)
                if dev == 'gpu':
                    self.io_buffers[dev][i]= cl.Buffer(ctx, cl.mem_flags.READ_WRITE,
                                                    size=self.data['io'][i][eo:eo + ne].nbytes)
                else:
                    self.io_buffers[dev][i]= cl.Buffer(ctx, cl.mem_flags.READ_WRITE|cl.mem_flags.USE_HOST_PTR,
                                                    hostbuf=self.data['io'][i][eo:eo+ne])


    def set_kernel_args(self, obj):
        """
        Sets Kernel Arguments (Buffers and Variable Arguments).

        :param obj: Device Type (cpu or gpu)
        :type obj: String
        """
        for i in range(len(self.input_buffers[obj])):
            self.kernel_objects[obj].set_arg(\
            self.buffer_info['input'][i]['pos'], \
            self.input_buffers[obj][i])
        for i in range(len(self.output_buffers[obj])):
            self.kernel_objects[obj].set_arg(self.buffer_info['output'][i]['pos'], self.output_buffers[obj][i])
        
        for i in range(len(self.io_buffers[obj])):
            
            self.kernel_objects[obj].set_arg(self.buffer_info['io'][i]['pos'], self.io_buffers[obj][i])
        for i in range(len(self.variable_args)):

            if type(self.variable_args[i]['value']) is list:
                self.kernel_objects[obj].set_arg(self.variable_args[i]['pos'],
                                                 make_ctype(self.variable_args[i]['type'])(
                                                     *self.variable_args[i]['value']))
            else:
                self.kernel_objects[obj].set_arg(self.variable_args[i]['pos'],
                                                 make_ctype(self.variable_args[i]['type'])(
                                                     self.variable_args[i]['value']))

        for i in range(len(self.local_args)):
            self.kernel_objects[obj].set_arg(self.local_args[i]['pos'],
                                             cl.LocalMemory(make_ctype(self.local_args[i]['type'])().nbytes * (
                                                 self.local_args[i]['size'])))

    def enqueue_write_buffers(self, queue, q_id, obj, size_percent=100, offset_percent=0, deps=None, **kwargs):
        """
        Enqueues list of write buffer operations to the OpenCL Runtime.

        :param queue: Command Queue for a CPU or a GPU device
        :type queue: pyopencl.CommandQueue
        :param q_id: ID of queue
        :type q_id: Integer
        :param obj: Device Type (CPU or GPU)
        :type obj: String
        :param size_percent: Percentage of problem space to be processed
        :type size_percent: Integer
        :param offset_percent: Percentage for offset required for selection of host array space to be copied to buffer
        :type offset_percent: Integer
        :param deps: Initial PyOpenCL Event on which subsequent write operations will be dependent stored in a list
        :type deps: list of pyopencl.Event Objects
        :param kwargs:
        :type kwargs:
        :return: Barrier Event signifying end of write operation
        :rtype: pyopencl.event
        """

        iev, ioev = [None] * len(self.input_buffers[obj]), [None] * len(self.io_buffers[obj])
        depends = [None] * (len(self.input_buffers[obj]) + len(self.io_buffers[obj]))
        if len(depends) == 0:
            depends = [None]
        if deps:
            depends[0] = deps
        logging.debug("PARTITION : Enqueuing Write Buffers %s",obj)

        start_barrier_event = cl.enqueue_barrier(queue, wait_for=depends[0])
        for i in range(len(self.input_buffers[obj])):
            if self.buffer_info['input'][i]['enq_write']:
                eo, ne = self.get_slice_values(self.buffer_info['input'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s : %s Enqueue_Write_Input_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s : %s Enqueue_Write_Input_Buffers_number_of_elements : %s", obj, self.name,
                              ne)
                if not kwargs['host_event'].write_start:
                    kwargs['host_event'].write_start = time.time()
                iev[i] = cl.enqueue_copy(queue, self.input_buffers[obj][i], self.data['input'][i][eo:eo + ne],
                                         is_blocking=False, wait_for=depends[i])
        # if self.input_buffers[obj]:
        #     depends = [None] * len(self.io_buffers[obj])
        j = len(self.input_buffers[obj])
        for i in range(len(self.io_buffers[obj])):
            if self.buffer_info['io'][i]['enq_write']:
                eo, ne = self.get_slice_values(self.buffer_info['io'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s : %s Enqueue_Write_IO_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s : %s Enqueue_Write_IO_number_of_elements : %s", obj, self.name, ne)

                if not kwargs['host_event'].write_start:
                    kwargs['host_event'].write_start = time.time()
                ioev[i] = cl.enqueue_copy(queue, self.io_buffers[obj][i], self.data['io'][i][eo:eo + ne],
                                          is_blocking=False, wait_for=depends[i + j])
        iev.extend(ioev)
        logging.debug("PARTITION : Number of write buffers %d" % (len(iev)))
        barrier_event = cl.enqueue_barrier(queue, wait_for=iev)

        if not self.lw_callback:
            barrier_event.set_callback(cl.command_execution_status.COMPLETE,
                                   notify_callback(self, obj, q_id, 'WRITE', iev ,host_event_info=kwargs['host_event']))
        return barrier_event

    def enqueue_nd_range_kernel(self, queue, q_id, obj, size_percent=100, offset_percent=0, deps=None, **kwargs):
        """
        Enqueues ND Range Kernel Operation for a kernel

        :param queue: OpenCL Command Queue
        :type queue: pyopencl.CommandQueue
        :param q_id: Framework specific id assigned to Command Queue
        :type q_id: Integer
        :param obj: OpenCL Device Type
        :type obj: String
        :param size_percent: Percentage of problem space to be processed
        :type size_percent: Integer
        :param offset_percent: Percentage for offset required for selection of host array space to be copied to buffer
        :type offset_percent: Integer
        :param deps:  PyOpenCL Event on which subsequent ndrange operations will be dependent on stored in a list
        :type deps: list of pyopencl.Event Objects
        :param kwargs:
        :type kwargs:
        :return: Barrier Event signifying end of ndrange operation
        :rtype: pyopencl.event
        """
        eo, ne = self.get_slice_values(self.buffer_info['input'][i], size_percent, offset_percent, **kwargs)

        global_work_offset = [0] * len(self.global_work_size)
        global_work_offset[0] =  eo

        if self.global_work_offset:
            global_work_offset = deepcopy(self.global_work_offset)

        global_work_size = deepcopy(self.global_work_size)
        local_work_size = deepcopy(self.local_work_size)

        global_work_size[0] = ne

        if 'C' in kwargs:
            coarsening_factor = kwargs['C']
            logging.debug("PARTITION: Coarsening Factor %d" % coarsening_factor)
            print "PYSCHEDCL", global_work_size,local_work_size, coarsening_factor
            global_work_size[0]/=coarsening_factor
            if local_work_size :
                local_work_size[0]/=coarsening_factor

        # global_work_offset[0] = multiple_round(global_work_size[0], offset_percent,self.partition_multiples, **kwargs)
        depends = [None]

        if deps:
            depends[0] = deps
        ev = None
        kwargs['host_event'].ndrange_start = time.time()
        if self.local_work_size:
            print "Dispatching",global_work_size,local_work_size,global_work_size[0]/local_work_size[0]
            ev = cl.enqueue_nd_range_kernel(queue, self.kernel_objects[obj], global_work_size,
                                            local_work_size, wait_for=depends[0],global_work_offset = global_work_offset)
        else:
            ev = cl.enqueue_nd_range_kernel(queue, self.kernel_objects[obj], global_work_size, None,
                                        wait_for=depends[0],global_work_offset = global_work_offset)


        barrier_event = cl.enqueue_barrier(queue, wait_for=[ev])

        if not self.lw_callback:
            barrier_event.set_callback(cl.command_execution_status.COMPLETE,
                                   notify_callback(self, obj, q_id, 'KERNEL', [ev],
                                                   host_event_info=kwargs['host_event']))
        return barrier_event

    def enqueue_read_buffers(self, queue, q_id, obj, size_percent=100, offset_percent=0, deps=None, callback=blank_fn,
                             **kwargs):
        """
        Enqueue Read Buffer operations for a kernel.

        :param queue: OpenCL Command Queue
        :type queue: pyopencl.CommandQueue
        :param q_id: Framework specific id assigned to Command Queue
        :type q_id: Integer
        :param obj: OpenCL Device Type
        :type obj: String
        :param size_percent: Percentage of problem space to be processed
        :type size_percent: Integer
        :param offset_percent: Percentage for offset required for selection of host array space to be copied to buffer
        :type offset_percent: Integer
        :param deps:  PyOpenCL Event on which subsequent ndrange operations will be dependent on stored in a list
        :type deps: list of pyopencl.Event Objects
        :param callback: Custom callback function
        :type callback: python function
        :param kwargs:
        :type kwargs:
        :return: Barrier Event signifying end of read operation
        :rtype: pyopencl.event
        """

        oev, ioev = [None] * len(self.output_buffers[obj]), [None] * len(self.io_buffers[obj])
        logging.debug("PARTITION : Enqueuing Read Buffers %s",obj)
        depends = [None] * (len(self.output_buffers[obj]) + len(self.io_buffers[obj]))
        if len(depends) == 0:
            depends = [None]
        if deps:
            depends[0] = deps
        kwargs['host_event'].read_start = time.time()
        for i in range(len(self.output_buffers[obj])):
            if self.buffer_info['output'][i]['enq_read']:
                eo, ne = self.get_slice_values(self.buffer_info['output'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s : %s Enqueue_Read_Output_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s : %s Enqueue_Read_Output_Buffers_number_of_elements : %s", obj, self.name,
                              ne)
                oev[i] = cl.enqueue_copy(queue, self.data['output'][i][eo:eo + ne], self.output_buffers[obj][i],
                                         is_blocking=False, wait_for=depends[i])
        j = len(self.output_buffers[obj])
        for i in range(len(self.io_buffers[obj])):
            if self.buffer_info['io'][i]['enq_read']:
                eo, ne = self.get_slice_values(self.buffer_info['io'][i], size_percent, offset_percent, **kwargs)
                logging.debug("PARTITION_%s : %s Enqueue_Read_IO_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s : %s Enqueue_Read_IO_Buffers_number_of_elements : %s", obj, self.name, ne)
                ioev[i] = cl.enqueue_copy(queue, self.data['io'][i][eo:eo + ne], self.io_buffers[obj][i],
                                          is_blocking=False, wait_for=depends[i + j])
        oev.extend(ioev)
        logging.debug("PARTITION : Number of read buffers %d" % (len(oev)))
        barrier_event = cl.enqueue_barrier(queue, wait_for=oev)
        barrier_event.set_callback(cl.command_execution_status.COMPLETE,
                                   notify_callback(self, obj, q_id, 'READ', oev,host_event_info=kwargs['host_event'],
                                                   callback=callback))
        return barrier_event

    def dispatch(self, gpu, cpu, ctxs, cmd_qs, lw_callback = False ,dep=None, partition=None, callback=blank_fn, **kwargs):
        """
        Dispatches Kernel with given partition class value (0,1,2,...,10). 0 is for complete CPU and 10 is for complete GPU.

        :param gpu: Denotes the index of gpu device in cmd_qs['gpu'] list or is -1 if we don't want to use device of this type.
        :type gpu: Integer
        :param cpu: Denotes the index of cpu device in cmd_qs['cpu'] list or is -1 if we don't want to use device of this type.
        :type cpu: Integer
        :param ctxs: Dictionary of Contexts for CPU and GPU devices.
        :type ctxs: dict
        :param cmd_qs: Dictionary of list of Command Queues
        :type cmd_qs: dict
        :param lw_callback : Set to true if callback should be initiated only for WRITE for each command queue on each device
        :type : bool
        :param dep: PyOpenCL Event on which subsequent write operations will be dependent on stored in a list
        :param partition: Integer from 0 to 10 or None denoting partition class value.
        :type partition: Integer
        :param callback: A function that will run on the host side once the kernel completes execution on the device. Handle unexpected arguments.
        :return: Tuple with first element being the starting time (host side) of dispatch and second element being list of kernel events for both CPU and GPU devices
        :rtype: Tuple

        """

        dispatch_start = time.time()
        logging.debug("DISPATCH : Dispatch function call for %s starts at %s", self.name, dispatch_start)

        gpu_host_events = HostEvents(self.name, self.id, dispatch_start=dispatch_start)
        cpu_host_events = HostEvents(self.name, self.id, dispatch_start=dispatch_start)


        self.lw_callback = lw_callback
        self.cb_events = {}

        if not lw_callback:
            for key in cmd_qs:
                n = len(cmd_qs[key])
                self.cb_events[key] = []
                for _ in range(n):
                    self.cb_events[key].append({"write_done": threading.Event(),"kernel_done" : threading.Event()})



        if partition is not None:
            self.partition = partition
        if dep:
            deps = dep
        else:
            deps = {key: cl.UserEvent(ctxs[key]) for key in ['cpu', 'gpu']}
        if gpu != -1 and cpu != -1:
            size_percent = self.partition * 10
        elif gpu == -1 and cpu != -1:
            size_percent = 0
            self.partition = 0
        elif cpu == -1 and gpu != -1:
            size_percent = 100
            self.partition = 10
        else:
            return None, None
        gdone, cdone = [], []
        if self.partition not in [0, 10]:
            self.chunks_left = 2
        if gpu != -1 and self.partition != 0:
            dispatch_id = generate_unique_id()

            # while test_and_set(0, 1):
            #     pass
            rqLock.acquire()
            global nGPU
            nGPU -= 1
            #rqlock[0] = 0
            rqLock.release()
            offset_percent = 0
            logging.debug("DISPATCH_gpu : Evaluation of kernel arguments for %s on GPU", self.name)
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent)
            gpu_host_events.create_buf_start = time.time()
            logging.debug("DISPATCH_gpu : Creation of buffers for %s on GPU", self.name)
            self.create_buffers(ctxs['gpu'], 'gpu', size_percent, offset_percent)
            gpu_host_events.create_buf_end = time.time()
            logging.debug("DISPATCH_gpu : Setting kernel arguments for %s on GPU", self.name)
            self.set_kernel_args('gpu')
            logging.debug("DISPATCH_gpu :Calling enqueue_write_buffers for GPU")
            gdone.append(self.enqueue_write_buffers(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, offset_percent,
                                                    deps=[deps['gpu']], did=dispatch_id, host_event=gpu_host_events))
            logging.debug("DISPATCH_gpu : Calling enqueue_nd_range_kernel for GPU")
            gdone.append(
                self.enqueue_nd_range_kernel(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, 0, deps=[gdone[-1]],
                                             did=dispatch_id, host_event=gpu_host_events,C=kwargs['C_gpu']))
            logging.debug("DISPATCH_gpu : Calling enqueue_read_buffers for GPU")
            gdone.append(self.enqueue_read_buffers(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, offset_percent,
                                                   deps=[gdone[-1]], callback=callback, did=dispatch_id,
                                                   host_event=gpu_host_events))

        if cpu != -1 and self.partition != 10:
            # while test_and_set(0, 1):
            #     pass
            rqLock.acquire()
            global nCPU
            nCPU -= 1
            #rqlock[0] = 0
            rqLock.release()
            dispatch_id = generate_unique_id()
            offset_percent = size_percent
            size_percent = 100 - size_percent
            logging.debug("DISPATCH_cpu : Evaluation of kernel arguments for %s on CPU", self.name)
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent)
            logging.debug("DISPATCH_cpu : Calling creating_buffers for %s on CPU", self.name)
            cpu_host_events.create_buf_start = time.time()
            self.create_buffers(ctxs['cpu'], 'cpu', size_percent, offset_percent)
            cpu_host_events.create_buf_end = time.time()
            logging.debug("DISPATCH_cpu : Calling set_kernel_args for %s on CPU", self.name)
            self.set_kernel_args('cpu')
            logging.debug("DISPATCH_cpu : Calling enqueue_write_buffers for %s on CPU", self.name)
            cdone.append(self.enqueue_write_buffers(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, offset_percent,
                                                    deps=[deps['cpu']], did=dispatch_id, host_event=cpu_host_events))
            logging.debug("DISPATCH_cpu : Evaluation of enqueue_nd_range_kernel for %s on CPU", self.name)
            cdone.append(
                self.enqueue_nd_range_kernel(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, 0, deps=[cdone[-1]],
                                             did=dispatch_id, host_event=cpu_host_events, C=kwargs['C_cpu']))
            logging.debug("DISPATCH_cpu : Evaluation of enqueue_read_buffers for %s on CPU", self.name)
            cdone.append(self.enqueue_read_buffers(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, offset_percent,
                                                   deps=[cdone[-1]], callback=callback, did=dispatch_id,
                                                   host_event=cpu_host_events))

        if not dep:
            for key in ['gpu', 'cpu']:
                deps[key].set_status(cl.command_execution_status.COMPLETE)

            start_time = time.time()
            logging.debug("DISPATCH : %s ke.dispatch_end %s ", self.name, start_time)
            logging.debug("DISPATCH : Evaluation of kernel arguments for %s ", self.name)
        logging.debug("DISPATCH : Number of events %d" % (len(gdone + cdone)))
        cmd_qs['gpu'][gpu].flush()
        cmd_qs['cpu'][cpu].flush()
        dispatch_end = time.time()
        logging.debug("DISPATCH : Dispatch function call for %s ends at %s", self.name, dispatch_end)
        return start_time, gdone + cdone

    ###########################################TODO#########################################################




    def enqueue_write_buffers_dag(self, task , h , queue, q_id, obj, n_chunks=1, chunk_number=0, deps=None, **kwargs):
        """
        Enqueues list of write buffer operations to the OpenCL Runtime.

        :param queue: Command Queue for a CPU or a GPU device
        :type queue: pyopencl.CommandQueue
        :param q_id: ID of queue
        :type q_id: Integer
        :param obj: Device Type (CPU or GPU)
        :type obj: String
        :param size_percent: Percentage of problem space to be processed
        :type size_percent: Integer
        :param offset_percent: Percentage for offset required for selection of host array space to be copied to buffer
        :type offset_percent: Integer
        :param deps: Initial PyOpenCL Event on which subsequent write operations will be dependent stored in a list
        :type deps: list of pyopencl.Event Objects
        :param kwargs:
        :type kwargs:
        :return: Barrier Event signifying end of write operation
        :rtype: pyopencl.event
        """

        iev = list()
        ioev =list()
        global enque_write , duplicate_write
        depends = [None] * (len(self.input_buffers[obj]) + len(self.io_buffers[obj]))
        if len(depends) == 0:
            depends = [None]
        if deps:
            depends[0] = deps
        logging.debug("PARTITION : Enqueuing Write Buffers %s",obj)
        #kwargs['host_event'].write_start = time.time()
        start_barrier_event = cl.enqueue_barrier(queue, wait_for=depends[0])
        for i in range(len(self.input_buffers[obj])):
            if self.buffer_info['input'][i]['enq_write']:
                if self.buffer_info['input'][i]['break'] ==0:
                    if chunk_number > 0:
                        continue
                    eo,ne = 0,self.buffer_info['input'][i]['size']
                else:
                    eo, ne = self.get_chunking_indices(self.buffer_info['input'][i]['size'], chunk_number, n_chunks)
                logging.debug("PARTITION_%s : %s Enqueue_Write_Input_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s : %s Enqueue_Write_Input_Buffers_number_of_elements : %s", obj, self.name,
                              ne)
                iev.append( cl.enqueue_copy(queue, self.input_buffers[obj][i], self.data['input'][i][eo:eo + ne],
                                         is_blocking=False, wait_for=depends[i],\
                                         device_offset=(self.data['input'][i][:eo].nbytes) ))

                #logging.info("CHECKING COPYING OF DATA")
                #print "ALL INPUT ",self.data['input'][i]
                #print "INPUT BEING COPIED",self.data['input'][i][eo:eo+ne]
                duplicate_write =  duplicate_write + 1
            enque_write = enque_write + 1

        # if self.input_buffers[obj]:
        #     depends = [None] * len(self.io_buffers[obj])

        j = len(self.input_buffers[obj])
        for i in range(len(self.io_buffers[obj])):
            if self.buffer_info['io'][i]['enq_write']:
                if self.buffer_info['io'][i]['break'] ==0:
                    if chunk_number > 0:
                        continue
                    eo,ne = 0,self.buffer_info['io'][i]['size']
                else:
                    eo, ne = self.get_chunking_indices(self.buffer_info['io'][i]['size'], chunk_number, n_chunks)
                logging.debug("PARTITION_%s : %s Enqueue_Write_IO_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s : %s Enqueue_Write_IO_number_of_elements : %s", obj, self.name, ne)
                logging.debug(str(depends)+"here")
                logging.debug(str(i+j))
                ioev.append( cl.enqueue_copy(queue, self.io_buffers[obj][i], self.data['io'][i][eo:eo + ne],
                                          is_blocking=False, wait_for=depends[i + j],\
                                          device_offset=self.data['io'][i][:eo].nbytes))

        iev.extend(ioev)
        logging.debug("PARTITION : Number of write buffers %d" % (len(iev)))
        if(len(iev) == 0):
            barrier_event = cl.enqueue_barrier(queue, wait_for=None)
        else:
            barrier_event = cl.enqueue_barrier(queue, wait_for=iev)
        # if iev:
        #     iev[-1].set_callback(cl.command_execution_status.COMPLETE,
        #                             notify_callback_dag(self , obj, q_id, 'WRITE', iev, host_event_info=kwargs['host_event'] ))
        #
        # else:
        #     barrier_event.set_callback(cl.command_execution_status.COMPLETE,
        #                             notify_callback_dag(self , obj, q_id, 'WRITE', iev, host_event_info=kwargs['host_event'] ))

        self.write_events.extend(iev)
        return barrier_event

    def enqueue_nd_range_kernel_dag(self,task , h , queue, q_id, obj, n_chunks=1, chunk_number=0, deps=None,cb = False ,**kwargs):
        """
        Enqueues ND Range Kernel Operation for a kernel

        :param queue: OpenCL Command Queue
        :type queue: pyopencl.CommandQueue
        :param q_id: Framework specific id assigned to Command Queue
        :type q_id: Integer
        :param obj: OpenCL Device Type
        :type obj: String
        :param size_percent: Percentage of problem space to be processed
        :type size_percent: Integer
        :param offset_percent: Percentage for offset required for selection of host array space to be copied to buffer
        :type offset_percent: Integer
        :param deps:  PyOpenCL Event on which subsequent ndrange operations will be dependent on stored in a list
        :type deps: list of pyopencl.Event Objects
        :param kwargs:
        :type kwargs:
        :return: Barrier Event signifying end of ndrange operation
        :rtype: pyopencl.event
        """
        #eo, ne = self.get_slice_values(self.buffer_info['input'][i], size_percent, offset_percent, **kwargs)

        global_work_offset = [0] * len(self.global_work_size)


        global_work_size = deepcopy(self.global_work_size)

        global_work_offset[0],global_work_size[0] = self.get_chunking_indices(global_work_size[0],chunk_number,n_chunks)
        #global_work_size[0] = multiple_round(global_work_size[0], size_percent,self.partition_multiples, **kwargs)


        #logging.info("OFFSET PERCENTAGE "+str(offset_percent))
        logging.debug("NDRANGE GLOBAL size and OFFSET "+str(global_work_size)+" "+str(global_work_offset))

        if deps:
            dependency = deps
        else:
            dependency = None


        kwargs['host_event'].ndrange_start = time.time()
        if self.local_work_size:
            ev = cl.enqueue_nd_range_kernel(queue, self.kernel_objects[obj], global_work_size,
                                            self.local_work_size, wait_for=dependency,global_work_offset = global_work_offset)
        else:
            ev = cl.enqueue_nd_range_kernel(queue, self.kernel_objects[obj], global_work_size, None,
                                            wait_for=dependency,global_work_offset = global_work_offset)

        barrier_event = cl.enqueue_barrier(queue, wait_for=[ev])
        # print "printing status of ndrange barrier ",barrier_event.get_info(cl.event_info.COMMAND_EXECUTION_STATUS)
        # print cl.command_execution_status.COMPLETE,cl.command_execution_status.RUNNING,cl.command_execution_status.SUBMITTED,cl.command_execution_status.QUEUED
        # # ev.set_callback(cl.command_execution_status.COMPLETE,
        #                            notify_callback_dag(self, obj ,q_id, 'KERNEL', [ev],
        #                                            host_event_info=kwargs['host_event'] ))

        # print "printing status of ndrange barrier ",barrier_event.get_info(cl.event_info.COMMAND_EXECUTION_STATUS)
        self.nd_range_event.append(ev)

        if cb:
            ev.set_callback(cl.command_execution_status.COMPLETE,
                                   notify_callback_dag(self,  obj, q_id, 'READ', [ev], host_event_info=kwargs['host_event'],
                                                   ))

        # self.nd_range_barrier = barrier_event

        return barrier_event

    def enqueue_read_buffers_dag(self,task ,h ,  queue, q_id, obj, n_chunks=1, chunk_number=0, deps=None, callback=blank_fn,
                             read_cb=True,**kwargs):
        """
        Enqueue Read Buffer operations for a kernel.

        :param queue: OpenCL Command Queue
        :type queue: pyopencl.CommandQueue
        :param q_id: Framework specific id assigned to Command Queue
        :type q_id: Integer
        :param obj: OpenCL Device Type
        :type obj: String
        :param size_percent: Percentage of problem space to be processed
        :type size_percent: Integer
        :param offset_percent: Percentage for offset required for selection of host array space to be copied to buffer
        :type offset_percent: Integer
        :param deps:  PyOpenCL Event on which subsequent ndrange operations will be dependent on stored in a list
        :type deps: list of pyopencl.Event Objects
        :param callback: Custom callback function
        :param kwargs:
        :type callback: python function
        :type kwargs:
        :return: Barrier Event signifying end of read operation
        :rtype: pyopencl.event
        """
        
        oev = list()
        ioev = list()
        global enque_read , duplicate_read
        ##print "oev , ioev" , oev , ioev
        logging.debug("PARTITION : Enqueuing Read Buffers %s",obj)
        depends = [None] * (len(self.output_buffers[obj]) + len(self.io_buffers[obj]))
        if len(depends) == 0:
            depends = [None]
        if deps:
            depends[0] = deps
        kwargs['host_event'].read_start = time.time()
        for i in range(len(self.output_buffers[obj])):
            if self.buffer_info['output'][i]['enq_read']:
                if self.buffer_info['output'][i]['break'] ==0:
                    if chunk_number > 0:
                        continue
                    eo,ne = 0,self.buffer_info['output'][i]['size']
                else:
                    eo, ne = self.get_chunking_indices(self.buffer_info['output'][i]['size'], chunk_number, n_chunks)
                logging.debug("PARTITION_%s : %s Enqueue_Read_Output_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s : %s Enqueue_Read_Output_Buffers_number_of_elements : %s", obj, self.name,
                              ne)
                logging.info("Enqueueing Read event for "+ str(self.name))
                oev.append(cl.enqueue_copy(queue, self.data['output'][i][eo:eo + ne], self.output_buffers[obj][i],
                                         is_blocking=False, wait_for=depends[i],device_offset=self.data['output'][i][:eo].nbytes))

                duplicate_read = duplicate_read + 1
            enque_read = enque_read + 1
        j = len(self.output_buffers[obj])
        for i in range(len(self.io_buffers[obj])):
            if self.buffer_info['io'][i]['enq_read']:
                if self.buffer_info['io'][i]['break'] ==0:
                    if chunk_number > 0 :
                        return
                    eo,ne = 0,self.buffer_info['io'][i]['size']
                else:
                    eo, ne = self.get_chunking_indices(self.buffer_info['io'][i]['size'], chunk_number, n_chunks)
                logging.debug("PARTITION_%s : %s Enqueue_Read_IO_Buffers_element_offset : %s", obj, self.name, eo)
                logging.debug("PARTITION_%s : %s Enqueue_Read_IO_Buffers_number_of_elements : %s", obj, self.name, ne)
                ioev.append(cl.enqueue_copy(queue, self.data['io'][i][eo:eo + ne], self.io_buffers[obj][i],
                                          is_blocking=False, wait_for=depends[i + j],device_offset=self.data['io'][i][:eo].nbytes))

        oev.extend(ioev)
        logging.debug("PARTITION : Number of read buffers %d" % (len(oev)))
        if(len(oev) == 0):
            barrier_event = cl.enqueue_barrier(queue, wait_for=None)
        else:
            barrier_event = cl.enqueue_barrier(queue, wait_for=oev)

        if oev and read_cb:
            oev[-1].set_callback(cl.command_execution_status.COMPLETE,
                                   notify_callback_dag(self,  obj, q_id, 'READ', oev, host_event_info=kwargs['host_event'],
                                                   callback=callback))

        # else:
        # barrier_event.set_callback(cl.command_execution_status.COMPLETE,
        #                            notify_callback_dag(self,  obj, q_id, 'READ', oev, host_event_info=kwargs['host_event'],
        #                                            callback=callback))
        self.read_events.extend(oev)
        return barrier_event

    def dispatch_dag(self,task , h,  gpu, cpu, ctxs, cmd_qs,n_chunks = 1,exec_dep=None, partition=10, callback=blank_fn, read_cb = True):
        """
        Dispatches Kernel with given partition class value (0,1,2,...,10). 0 is for complete CPU and 10 is for complete GPU.

        :param gpu: Denotes the index of gpu device in cmd_qs['gpu'] list or is -1 if we don't want to use device of this type.
        :type gpu: Integer
        :param cpu: Denotes the index of cpu device in cmd_qs['cpu'] list or is -1 if we don't want to use device of this type.
        :type cpu: Integer
        :param ctxs: Dictionary of Contexts for CPU and GPU devices.
        :type ctxs: dict
        :param cmd_qs: Dictionary of list of Command Queues
        :type cmd_qs: dict
        :param dep: PyOpenCL Event on which subsequent write operations will be dependent on stored in a list
        :param partition: Integer from 0 to 10 or None denoting partition class value.
        :type partition: Integer
        :param callback: A function that will run on the host side once the kernel completes execution on the device. Handle unexpected arguments.
        :return: Tuple with first element being the starting time (host side) of dispatch and second element being list of kernel events for both CPU and GPU devices
        :rtype: Tuple

        """

        assert partition in [0,10], "DAG dispatch doesnt allow partitioning, only supports single device"

        dispatch_start = time.time()
        logging.debug("DISPATCH : Dispatch function call for %s starts at %s", self.name, dispatch_start)





        self.partition = partition

        # if dep:
        #     deps = dep
        # else:
        #     deps = {key: cl.UserEvent(ctxs[key]) for key in ['cpu', 'gpu']}
        #
        user_deps = {key: cl.UserEvent(ctxs[key]) for key in ['cpu', 'gpu']}


        gdone, cdone = [], []



        if self.local_work_size:
            ###HARD CODED FOR N CHUNKS =1, remove these comments if you want chunking
            # assert self.global_work_size[0]%self.local_work_size[0] == 0, "Global work size {} should be a multiple of local work size {}".format(self.global_work_size[0],self.local_work_size[0])
            # assert self.check_pow_2(self.global_work_size[0]) and self.check_pow_2(self.local_work_size[0]), "Global and local work group sizes should be a power of 2"

            max_chunk_number = self.global_work_size[0]//self.local_work_size[0]
            corrected_n_chunks = max_chunk_number
            while corrected_n_chunks > n_chunks:
                corrected_n_chunks = corrected_n_chunks//2

            if corrected_n_chunks != n_chunks:
                logging.debug("Changing number of chunks to accomodate local work group size")
            n_chunks = corrected_n_chunks

        #####HARD CODING FOR NUM CHUNKS = 1#############
        n_chunks = 1

        host_events = [HostEvents(self.name, self.id, dispatch_start=dispatch_start) for _ in range(n_chunks)]
        self.host_events = host_events


        if self.partition == 10:
            dispatch_id = generate_unique_id()

            rqLock.acquire()
            global nGPU
            nGPU -= 1
            rqLock.release()

            offset_percent = 0
            size_percent = 100

            logging.debug("DISPATCH_gpu : Evaluation of kernel arguments for %s on GPU", self.name)
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent)

            for i in range(n_chunks):
                host_events[i].create_buf_start = time.time()

            logging.debug("DISPATCH_gpu : Creation of buffers for %s on GPU", self.name)
            self.create_buffers2(ctxs['gpu'], 'gpu', size_percent, offset_percent)

            for i in range(n_chunks):
                host_events[i].create_buf_end = time.time()

            logging.debug("DISPATCH_gpu : Setting kernel arguments for %s on GPU", self.name)
            self.set_kernel_args('gpu')

            ###handle chunks
            ##calculate chunk size

            host_events[0].write_start = time.time()
            for chunk_number in range(n_chunks):

                logging.debug("DISPATCH_gpu :Calling enqueue_write_buffers for GPU")
                logging.debug("OFFSET PERCENT "+str(offset_percent)+" SIZE PERCENT "+str(size_percent))

                gdone.append(
                            self.enqueue_write_buffers_dag(task ,h, cmd_qs['gpu'][gpu],\
                                                        gpu, 'gpu', n_chunks, chunk_number,
                                                        deps=[user_deps['gpu']], did=dispatch_id, \
                                                        host_event=host_events[chunk_number])
                            )

                logging.debug("DISPATCH_gpu : Calling enqueue_nd_range_kernel for GPU")


                gdone.append(
                    self.enqueue_nd_range_kernel_dag(task ,h,cmd_qs['gpu'][gpu],\
                                                    gpu, 'gpu', n_chunks, chunk_number,\
                                                    deps=exec_dep,\
                                                    did=dispatch_id, host_event=host_events[chunk_number]))



                offset_percent += size_percent
            logging.debug("DISPATCH_gpu : Calling enqueue_read_buffers for GPU")
            gdone.append(self.enqueue_read_buffers_dag(task , h, cmd_qs['gpu'][gpu], gpu, 'gpu', n_chunks,\
                                                   chunk_number=0,deps=[gdone[-1]], callback=callback,\
                                                   did=dispatch_id,
                                                   host_event=host_events[chunk_number],read_cb=read_cb))


        else:

            rqLock.acquire()
            global nCPU
            nCPU -= 1
            rqLock.release()

            dispatch_id = generate_unique_id()
            offset_percent = 0
            size_percent = 100

            logging.debug("DISPATCH_cpu : Evaluation of kernel arguments for %s on CPU", self.name)
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent)

            logging.debug("DISPATCH_cpu : Calling creating_buffers for %s on CPU", self.name)

            for i in range(n_chunks):
                host_events[i].create_buf_start = time.time()

            self.create_buffers2(ctxs['cpu'], 'cpu', size_percent, offset_percent)

            for i in range(n_chunks):
                host_events[i].create_buf_end = time.time()

            logging.debug("DISPATCH_cpu : Calling set_kernel_args for %s on CPU", self.name)
            self.set_kernel_args('cpu')


            offset_percent = 0

            host_events[0].write_start = time.time()

            if not exec_dep:
                exec_dep = [user_deps['cpu']]
            else:
                exec_dep = exec_dep+[user_deps['cpu']]

            for chunk_number in range(n_chunks):


                logging.debug("DISPATCH_cpu : Calling enqueue_write_buffers for %s on CPU", self.name)
                # cdone.append(self.enqueue_write_buffers_dag(task , h, cmd_qs['cpu'][cpu], cpu, 'cpu', n_chunks, chunk_number,
                #                                         deps=[user_deps['cpu']], did=dispatch_id, host_event=host_events[chunk_number]))

                ("DISPATCH_cpu : Evaluation of enqueue_nd_range_kernel for %s on CPU", self.name)

                cdone.append(
                    self.enqueue_nd_range_kernel_dag(task , h,  cmd_qs['cpu'][cpu], cpu, 'cpu', n_chunks, chunk_number, deps=exec_dep,
                                                did=dispatch_id, host_event=host_events[chunk_number],cb = read_cb))


                ("DISPATCH_cpu : Evaluation of enqueue_read_buffers for %s on CPU", self.name)

                offset_percent+=size_percent

            # cdone.append(self.enqueue_read_buffers_dag(task , h , cmd_qs['cpu'][cpu], cpu, 'cpu', n_chunks, 0,
            #                                        deps=[cdone[-1]], callback=callback, did=dispatch_id,
            #                                        host_event=host_events[chunk_number],read_cb=read_cb))

        #print "printing status of ndrange barrier ",self.nd_range_barrier.get_info(cl.event_info.COMMAND_EXECUTION_STATUS)
        # if not dep:
        # for key in ['gpu', 'cpu']:
        #     user_deps[key].set_status(cl.command_execution_status.COMPLETE)



        logging.debug("DISPATCH : Number of events %d" % (len(gdone + cdone)))
        #print "printing status of ndrange barrier ",self.nd_range_barrier.get_info(cl.event_info.COMMAND_EXECUTION_STATUS)

        # cmd_qs['gpu'][gpu].flush()
        # cmd_qs['cpu'][cpu].flush()

        #start_time = time.time()
        dispatch_end = time.time()
        #("DISPATCH : Dispatch function call for %s ends at %s", self.name, dispatch_end)
        #print "printing status of ndrange barrier ",self.nd_range_barrier.get_info(cl.event_info.COMMAND_EXECUTION_STATUS)
        return  gdone + cdone,user_deps

    ################################################################################################
    def dispatch_multiple(self, gpus, cpus, ctxs, cmd_qs, dep=None, partition=None, callback=blank_fn):
        """
        Dispatch Kernel across multiple devices  with given partition class value (0,1,2,...,10). 0 is for complete CPU and 10 is for complete GPU.

        :param gpus: A list of gpu device ids
        :type gpus: list
        :param cpus: A list of cpu device ids
        :type cpus: list
        :param ctxs: Dictionary of Contexts for CPU and GPU devices.
        :type ctxs: dict
        :param cmd_qs: Dictionary of list of Command Queues
        :type cmd_qs: dict
        :param dep: PyOpenCL Event on which subsequent write operations will be dependent on stored in a list
        :param partition: Integer from 0 to 10 or None denoting partition class value.
        :type partition: Integer
        :param callback: A function that will run on the host side once the kernel completes execution on the device. Handle unexpected arguments.
        :return: Tuple with first element being the starting time (host side) of dispatch and second element being list of kernel events for both CPU and GPU devices
        :rtype: Tuple

        """
        dispatch_start = time.time()
        logging.debug("DISPATCH : Dispatch function call for %s starts at %s", self.name, dispatch_start)
        # while test_and_set(0, 1):
        #     pass
        rqLock.acquire()
        global nGPU
        nGPU -= len(gpus)
        global nCPU
        nCPU -= len(cpus)
        #rqlock[0] = 0
        rqLock.release()
        self.multiple_break = True
        self.chunks_cpu += len(gpus)
        self.chunks_gpu += len(cpus)
        self.chunks_left = len(gpus) + len(cpus)
        if partition is not None:
            self.partition = partition
        total = len(cpus) + len(gpus)
        size_percent = 100 / total
        if len(gpus) != 0 and len(cpus) != 0:
            gpu_percent = self.partition * 10
        elif len(gpus) == 0 and len(cpus) != 0:
            gpu_percent = 0
            self.partition = 0
        elif len(cpus) == 0 and len(gpus) != 0:
            gpu_percent = 100
            self.partition = 10
        else:
            return None, None
        if gpu_percent == 0:
            nGPU += len(gpus)
        if gpu_percent == 100:
            nCPU += len(cpus)
        cpu_percent = 100 - gpu_percent
        rqlock[0] = 0
        gdone, cdone = [], []

        if dep:
            deps = dep
        else:
            deps = dict()
            deps['gpu'] = cl.UserEvent(ctxs['gpu'])
            deps['cpu'] = cl.UserEvent(ctxs['cpu'])
        if len(gpus) != 0:
            size_percent = gpu_percent / len(gpus)
        for i in range(len(gpus)):
            dispatch_id = generate_unique_id()
            gpu_host_events = HostEvents(self.name, self.id, dispatch_start=dispatch_start)
            offset_percent = size_percent * i
            exact = 1
            if i == total - 1:
                size_percent = 100 - offset_percent
                exact = 0
            gpu = gpus[i]
            logging.debug("DISPATCH_gpu : Evaluation of kernel arguments for %s on GPU", self.name)
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent, exact=exact)
            gpu_host_events.create_buf_start = time.time()
            logging.debug("DISPATCH_gpu : Calling creating_buffer for %s on GPU", self.name)
            self.create_buffers(ctxs['gpu'], 'gpu', size_percent, offset_percent, exact=exact)
            gpu_host_events.create_buf_end = time.time()
            self.set_kernel_args('gpu')
            logging.debug("DISPATCH_gpu : %s Calling enqueue_write_buffers for GPU", self.name)
            gdone.append(self.enqueue_write_buffers(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, offset_percent,
                                                    deps=[deps['gpu']], exact=exact, did=dispatch_id,
                                                    host_event=gpu_host_events))
            logging.debug("DISPATCH_gpu : %s Calling enqueue_nd_range_kernel for GPU", self.name)
            gdone.append(
                self.enqueue_nd_range_kernel(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, 0, deps=[gdone[-1]],
                                             exact=exact,
                                             did=dispatch_id, host_event=gpu_host_events))
            logging.debug("DISPATCH_gpu : %s Calling enqueue_read_buffers for GPU", self.name)
            gdone.append(
                self.enqueue_read_buffers(cmd_qs['gpu'][gpu], gpu, 'gpu', size_percent, offset_percent,
                                          deps=[gdone[-1]], exact=exact,
                                          callback=callback, did=dispatch_id, host_event=gpu_host_events))

        if len(cpus) != 0:
            size_percent = cpu_percent / len(cpus)
        for i in range(len(cpus)):
            dispatch_id = generate_unique_id()

            cpu_host_events = HostEvents(self.name, self.id, dispatch_start=dispatch_start)
            exact = 1
            offset_percent = size_percent * i + gpu_percent
            if i == total - 1 - len(gpus):
                size_percent = 100 - offset_percent
                exact = 0
            cpu = cpus[i]
            logging.debug("DISPATCH_cpu : Evaluation of kernel arguments for %s on CPU", self.name)
            self.eval_vargs(size_percent=size_percent, offset_percent=offset_percent, exact=exact)
            cpu_host_events.create_buf_start = time.time()
            logging.debug("DISPATCH_cpu : Calling creating_buffer for %s on CPU", self.name)
            self.create_buffers(ctxs['cpu'], 'cpu', size_percent, offset_percent, exact=exact)
            cpu_host_events.create_buf_end = time.time()
            self.set_kernel_args('cpu')
            logging.debug("DISPATCH_cpu : Calling enqueue_write_buffers for %s on CPU", self.name)
            cdone.append(self.enqueue_write_buffers(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, offset_percent,
                                                    deps=[deps['cpu']], exact=exact, did=dispatch_id,
                                                    host_event=cpu_host_events))
            logging.debug("DISPATCH_cpu : Evaluation of enqueue_nd_range_kernel for %s on CPU", self.name)
            cdone.append(
                self.enqueue_nd_range_kernel(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, 0, deps=[cdone[-1]],
                                             exact=exact,
                                             did=dispatch_id, host_event=cpu_host_events))
            logging.debug("DISPATCH_cpu : Evaluation of enqueue_read_buffers for %s on CPU", self.name)
            cdone.append(
                self.enqueue_read_buffers(cmd_qs['cpu'][cpu], cpu, 'cpu', size_percent, offset_percent,
                                          deps=[cdone[-1]], exact=exact,
                                          callback=callback, did=dispatch_id, host_event=cpu_host_events))

        if not dep:
            for key in ['gpu', 'cpu']:
                deps[key].set_status(cl.command_execution_status.COMPLETE)
            start_time = time.time()

        return start_time, gdone + cdone

    def get_device_requirement(self):

        req = {'gpu': 0, 'cpu': 0, 'all': 0}
        if self.partition > 0:
            req['gpu'] += 1
            req['all'] += 1
        if self.partition < 10:
            req['cpu'] += 1
            req['all'] += 1
        return req


def get_platform(vendor_name):
    """
    Gets platform given a vendor name

    :param vendor_name: Name of OpenCL Vendor
    :type vendor_name: string
    :return: OpenCL platform related with vendor name
    :rtype: PyOpenCL platform object
    """
    platforms = cl.get_platforms()
    if len(platforms):
        for pt in cl.get_platforms():
            if vendor_name in pt.name:
                return pt
        print(vendor_name + " Platform not found.")
    else:
        print("No platform found.")


def get_multiple_devices(platform, dev_type, num_devs):
    """
    Get Multiple Devices given a platform and dev type.

    :param platform: OpenCL Platform pertaining to some vendor
    :type platform: pyopencl.platform object
    :param dev_type: Device Type (CPU or GPU)
    :type dev_type: pyopencl.device_type object
    :param num_devs: Number of Devices
    :type num_devs: Integer
    :return: List of OpenCL devices
    :rtype: list of pyopencl.device objects
    """
    devs = platform.get_devices(device_type=dev_type)
    if num_devs > len(devs):
        print("Requirement: " + str(num_devs) + " greater than availability: " + str(len(devs)))
    else:
        return devs[:num_devs]


def get_single_device(platform, dev_type):
    """
    Get Single Device given a platform and dev type.

    :param platform: OpenCL platform pertaining to some vendor
    :type platform: pyopencl.platform object
    :param dev_type: Device Type (CPU or GPU)
    :type dev_type: pyopencl.device_type object
    :return: List containing one OpenCL device
    :rtype: List containing one pyopencl.device object
    """
    return get_multiple_devices(platform, dev_type, 1)


def get_sub_devices(platform, dev_type, num_devs, total_compute=32):
    """
    Get Sub Devices given a platform and dev type.

    :param platform: OpenCL platform pertaining to some vendor
    :type platform: pyopencl.platform object
    :param dev_type: Device Type (CPU or GPU)
    :type dev_type: pyopencl.device_type object
    :param num_devs: Number of devices
    :type num_devs: Integer
    :param total_compute: Total Number of Compute Units for an OpenCL device
    :type total_compute: Integer
    :return: List of OpenCL subdevices for a particular device
    :rtype: list of pyopencl.device objects
    """
    dev = get_single_device(platform, dev_type)[0]
    return dev.create_sub_devices([cl.device_partition_property.EQUALLY, total_compute / num_devs])


def create_command_queue_for_each(devs, ctx):
    """
    Creates command queue for a specified number of devices belonging to a context provided as argument

    :param devs: List of OpenCL devices
    :type devs: List of pyopencl.device objects
    :param ctx: OpenCL Context
    :rtype ctx: pyopencl.context object
    :return: List of OpenCL Command Queues
    :rtype: list of pyopencl.CommandQueue objects
    """
    cmd_qs = [cl.CommandQueue(ctx, device=dev, properties=cl.command_queue_properties.PROFILING_ENABLE \
                                ) for dev in devs]


    return cmd_qs


def create_multiple_command_queue_for_device(devs, ctx, num_queues):
    """
    Creates multiple command queue for a single device belonging to a context provided as argument

    :param devs: List of OpenCL devices
    :type devs: List of pyopencl.device objects
    :param ctx: OpenCL Context
    :rtype ctx: pyopencl.context object
    :return: List of OpenCL Command Queues
    :rtype: list of pyopencl.CommandQueue objects
    """
    assert len(devs) == 1
    cmd_qs = [cl.CommandQueue(ctx, device=devs[0], properties=cl.command_queue_properties.PROFILING_ENABLE \
                                ) for _ in range(num_queues)]


    return cmd_qs


def query_for_resources():
    platforms = cl.get_platforms()
    cpu_platforms = set()
    gpu_platforms = set()
    num_cpu_devices = 0
    num_gpu_devices = 0
    for platform in platforms:
        devices = platform.get_devices()
        for dev in devices:
            if cl.device_type.to_string(dev.type) == "GPU":
                gpu_platforms.add(platform.get_info(cl.platform_info.NAME))
                num_gpu_devices += 1
            if cl.device_type.to_string(dev.type) == "CPU":
                cpu_platforms.add(platform.get_info(cl.platform_info.NAME))
                num_cpu_devices += 1

    return num_cpu_devices,num_gpu_devices,list(cpu_platforms),list(gpu_platforms)


def host_initialize(num_gpus, num_cpus,use_mul_queues = False ,local=False):
    """
    Set local=True if your device doesn't support GPU. But you still
    want to pretend as if you have one.

    :param num_gpus: Number of GPU Devices
    :type num_gpus: Integer
    :param num_cpus: Number of CPU Devices
    :type num_cpus: Integer
    :param local: Flag for Specifying whether platform supports GPU or not
    :type local: Boolean
    :param cpu_platform: CPU Platform Name
    :type cpu_platform: String
    :param gpu_platform: GPU Platform Name
    :type gpu_platform: String
    :return: Returns a tuple comprising command queue dictionary, context dictionary and list of OpenCL CPU and GPU devices.
    :rtype: Tuple


    """
    _,_,cpu_platforms,gpu_platforms = query_for_resources()
    gpu_platform = gpu_platforms[0]
    cpu_platform = cpu_platforms[0]

    global nGPU
    global nCPU
    if local:
        gpus = None
        cpu_platform = get_platform(cpu_platform)
        if num_cpus > 1 and cl.get_cl_header_version() >= (1, 2):
            cpus = get_sub_devices(cpu_platform, cl.device_type.CPU, num_cpus, 4)
        else:
            cpus = get_single_device(cpu_platform, cl.device_type.CPU)
        ctx = cl.Context(devices=cpus)
        ctxs = {"gpu": ctx, "cpu": ctx}
        cmd_q = create_command_queue_for_each(cpus, ctxs['cpu'])
        # cmd_qs = {"gpu":cmd_q, "cpu": cmd_q}
        cmd_qs = {"gpu": [cmd_q[0]], "cpu": [cmd_q[1]]}
        ready_queue['gpu'].append(0)
        ready_queue['cpu'].append(0)
        device_history['gpu'].append([])
        device_history['cpu'].append([])
        gpus = [cpus[0]]
        cpus = [cpus[1]]
        nGPU = 1
        nCPU = 1

    else:
        gpus, cpus = [], []
        ctxs = {"gpu": None, "cpu": None}
        cmd_qs = {
            "gpu": [],
            "cpu": []
        }
        if num_gpus > 0:
            gpu_platform = get_platform(gpu_platform) #taken from cons.gpu_platform, checks if vendor is present in all platforms
            if not use_mul_queues:
                gpus = get_multiple_devices(gpu_platform, cl.device_type.GPU, num_gpus) #asks for number of devices requested
                global MAX_GPU_ALLOC_SIZE
                ctxs['gpu'] = cl.Context(devices=gpus)
                cmd_qs['gpu'] = create_command_queue_for_each(gpus, ctxs['gpu'])
            else:
                gpus = get_multiple_devices(gpu_platform, cl.device_type.GPU, 1)
                ctxs['gpu'] = cl.Context(devices=gpus)
                cmd_qs['gpu'] = create_multiple_command_queue_for_device(gpus, ctxs['gpu'],num_gpus)

        if num_cpus > 0:
            # cpu_platform = get_platform("AMD Accelerated Parallel Processing")
            cpu_platform = get_platform(cpu_platform)
            if num_cpus > 1 and cl.get_cl_header_version() >= (1, 2):
                if not use_mul_queues:
                    cpus = get_sub_devices(cpu_platform, cl.device_type.CPU, num_cpus)
                else:
                    cpus = get_single_device(cpu_platform, cl.device_type.CPU)
            else:
                cpus = get_single_device(cpu_platform, cl.device_type.CPU)
            # print cpus
            global MAX_CPU_ALLOC_SIZE
            MAX_CPU_ALLOC_SIZE = cpus[0].max_mem_alloc_size
            ctxs['cpu'] = cl.Context(devices=cpus)
            if use_mul_queues:
                cmd_qs['cpu'] = create_multiple_command_queue_for_device(cpus, ctxs['cpu'],num_cpus)
            else:
                cmd_qs['cpu'] = create_command_queue_for_each(cpus, ctxs['cpu'])
        nGPU = len(cmd_qs['gpu'])
        nCPU = len(cmd_qs['cpu'])
        for key in cmd_qs.keys():
            ready_queue[key].extend(range(len(cmd_qs[key])))
            device_history[key].extend([[] for i in range(len(cmd_qs[key]))])
    global system_cpus
    global system_gpus
    system_cpus = nCPU
    system_gpus = nGPU
    return cmd_qs, ctxs, gpus, cpus


def host_synchronize(cmd_qs, events):
    """
    Ensures that all operations in all command queues and associated events have finished execution

    :param cmd_qs: Dictionary of list of Command Queues
    :type cmd_qs: dict
    :param events: List of OpenCL Events
    :type events: list of pyopencl.events
    """
    global nCPU, nGPU, system_cpus, system_gpus
    while nCPU < system_cpus or nGPU < system_gpus:
        pass
    global callback_queue
    while any(callback_queue[key] == False for key in callback_queue.keys()):
        pass
    callback_queue.clear()
    for event in events:
        event.wait()
    for key in cmd_qs:
        for q in cmd_qs[key]:
            import sys
            # print "Command Queue Size",key, sys.getsizeof(q)
            q.flush()
            q.finish()


def build_kernel_from_info(info_file_name, gpus, cpus, ctxs):
    """
    Create Kernel object from info.

    :param info_file_name: Name of OpenCL Kernel Specification File (JSON)
    :type info_file_name: String
    :param gpus: List of OpenCL GPU devices
    :type gpus: list of pyopencl.device objects
    :param cpus: List of OpenCL CPU devices
    :type cpus: list of pyopencl.device objects
    :param ctxs: Dictionary of contexts keyed by device types
    :type ctxs: dict
    :return: Dictionary of key: value pairs where key is device type and value is the device specific binary compiled for the kernel
    :rtype: dict


    """
    import json
    info = json.loads(open(info_file_name).read())
    ki = Kernel(info)
    ki.build_kernel(gpus, cpus, ctxs)
    return ki


###################TODO###################################



######################TODO###################################

class kernelFlags(object):
    def __init__(self,parents,finished_parents = 0):
        self.parents = parents
        self.finished_parents = finished_parents

    def is_ready(self):
        return self.finished_parents == len(self.parents)

class Task(object):
    """
    Class to handle all operations performed on OpenCL task object.
    An OpenCL task object contains one or multiple OpenCL kernels.

    :ivar rank: Rank value denoting priority of task in DAG
    :ivar ECO: Estimated Computation Overhead of Task Object
    :ivar id: Unique identifier for task object
    :ivar dev_requirement: Number of devices required by task object for execution
    :ivar kernels: Set of kernel objects contained in the task object
    :ivar finished_kernels: Set of kernel objects that have finished execution
    :ivar free_kernels: List of kernel objects ready for dispatch
    :ivar processed_kernels: Set of kernels that have been dispatched
    :ivar processing_kernel: List of kernels currently executing


    """
    import operator

    def __init__(self, init_kernel,task_dag):
        """
        :param kernel:
        :type kernel: Kernel
        """
        import operator
        self.rank = 0
        self.rank_b = 0
        self.ECO = init_kernel.eco
        # self.id = generate_unique_id()
        self.id = init_kernel.id
        self.task_dag_object = task_dag
        self.dependencies = {}

        self.kernels = set()
        self.kernel_ids = set()

        self.dev_requirement = {'gpu': 0, 'cpu': 0, 'all': 0}
        self.free_devices = []
        self.occupied_devices = []
        self.allocated_devices = []
        self.resource_lock = threading.Condition()
        self.kernels.add(init_kernel)
        self.kernel_ids.add(init_kernel.id)

        self.finished_kernels = set()
        self.free_kernels = list()
        self.processed_kernels = set()
        self.recently_added_kernels = list()
        self.modify_device_requirement(self.kernels, operator.iadd)
        self.partition = init_kernel.partition
        self.processing_kernel = list()
        self.optm_device = init_kernel.optm_device
        if int(self.optm_device) == 10:
            self.device = "gpu"
        else:
            self.device = "cpu"

        self.to_be_schedule = False
        init_kernel.task_object = self

        #self.free_kernels_lock = threading.Lock()
        self.ready_to_dispatch = threading.Event()
        self.has_dispatched = False
        self.has_enqueued = False


    def set_device(self,optm_device):
        self.optm_device=optm_device
        if optm_device == 10:
            self.device = "gpu"
        else:
            self.device = "cpu"

    def allocate_devices(self,devices):
        logging.info("Allocating device " + str(self.device))
        rqLock.acquire()
        for dev_number in devices:
            ready_queue[self.device].remove(dev_number)
        rqLock.release()

        logging.debug("acquiring lock in allocate_devices")
        self.resource_lock.acquire()
        for dev_number in devices:
            self.free_devices.append(dev_number)
            self.allocated_devices.append(dev_number)
        self.resource_lock.release()
        logging.debug("releasing lock in allocate_devices")


    def set_partition(self, partition):
        self.partition = partition
        self.optm_device = partition
        if int(self.optm_device) == 10:
            self.device = "gpu"
        else:
            self.device = "cpu"

    def update_task_info(self,task_mappings, kernels , total_kernels):
        new = list()
        for j in kernels:
            if(total_kernels[j]):
                new.append(total_kernels[j])
        for i in self.kernels:
            if(i in new):
                self.free_kernels.append(i)

    def add_free_kernels(self,free_kernels):
        logging.debug("testing task.add_free_kernels ")
        # print "trying to add ",free_kernels
        # print "task kernels ", self.kernels
        logging.info("EVENT: Adding free kernels " + str([f.id for f in free_kernels]) + " to " + str([f.id for f in self.free_kernels]) +" of task" + str(self.id)) 
        for kernel in free_kernels:
            assert kernel in self.kernels

        self.free_kernels.extend(free_kernels)

    def add_finished_kernels(self,fin_kernels):
        logging.debug("adding finished kernel")
        for kernel in fin_kernels:
            assert kernel in self.kernels
            self.finished_kernels.add(kernel)


    def add_kernel(self, kernel):
        self.kernels.add(kernel)
        self.kernel_ids.add(kernel.id)
        kernel.task_object = self

    def load_dependent_data_and_buffers(self,kernel):
        """
        :param kernel_id:
        :type int
        :return:
        """

        dag = self.task_dag_object
        kernel_id = kernel.id
        dependencies = dag.get_kernel_parent_ids(kernel_id)

        #external_dependencies = set(dependencies) - set(self.get_kernel_ids())

        for key in ['input', 'io']:
            for i in range(len(kernel.buffer_info[key])):
                if 'from' in kernel.buffer_info[key][i]:
                    data_dep = kernel.buffer_info[key][i]['from']
                    if int(data_dep['kernel']) not in self.kernel_ids:
                        kernel.data[key][i] = dag.get_kernel(int(data_dep['kernel'])).get_data(int(data_dep['pos']))
                    elif int(data_dep['kernel']) in self.kernel_ids:
                        dependency = dag.get_kernel(int(data_dep['kernel']))
                        dbuff = dependency.get_buffer(int(data_dep['pos']))
                        if key is 'input':
                            kernel.input_buffers['gpu'][i] = dbuff['gpu']
                            kernel.input_buffers['cpu'][i] = dbuff['cpu']
                        elif key is 'io':
                            kernel.io_buffers['gpu'][i] = dbuff['gpu']
                            kernel.io_buffers['cpu'][i] = dbuff['cpu']
                        else:
                            raise

    def prepare_kernel(self, kid):
        """
        Prepares Kerneli with id kid by modifying properties of its buffers, so that buffers can be reused.
        :param kid:
        :param dag:
        :return:
        """
        dag = self.task_dag_object
        dependents = dag.get_kernel_children_ids(kid)
        kernel = dag.get_kernel(kid)
        internal_dependents = set(self.get_kernel_ids()) & set(dependents)
        # internal_dependents = set(dependents)
        #print "kid , internal_dependents",kid , internal_dependents
        for dep in internal_dependents:
            dependent_kernel = dag.get_kernel(dep)
            #print "dependent_kernel.buffer_deps" , dependent_kernel.buffer_deps , dep
            for i in dependent_kernel.buffer_deps:
                #print " dependent_kernel.buffer_deps[i][0] == kid ", dependent_kernel.buffer_deps[i][0] , kid , type(dependent_kernel.buffer_deps[i][0]) , type(kid)
                if int(dependent_kernel.buffer_deps[i][0]) == kid:
                    dep_btype, dep_loc = dependent_kernel.get_buffer_info_location(i)
                    pos = dependent_kernel.buffer_deps[i][1]
                    btype, j = kernel.get_buffer_info_location(pos)
                    kernel.buffer_info[btype][j]['enq_read'] = False
                    dependent_kernel.buffer_info[dep_btype][dep_loc]['create'] = False
                    dependent_kernel.buffer_info[dep_btype][dep_loc]['enq_write'] = False
                    
                    

    def prepare_kernels(self):
        """
        Prepares all kernels in dag by modifying properties of their buffers, so that buffers can be reused.
        :param dag:
        :return:
        """
        dag = self.task_dag_object
        for kernel in self.get_kernels_sorted(dag):
            self.prepare_kernel(kernel.id)

    def build_kernels(self, gpus, cpus, ctxs):
        """
        Build all kernels in the task.
        :param gpus:
        :param cpus:
        :param ctxs:
        :return:
        """
        for kernel in self.get_kernels():
            kernel.build_kernel(gpus, cpus, ctxs)



    def dispatch_single(self,gpus = None, cpus = None,  ctxs = None, cmd_qs = None,h=None, callback=blank_fn, *args, **kwargs):
        """
        Dispatch any of the free kernel of the task on the particular device
        :param gpu:
        :param cpu:
        :param ctxs:
        :param cmd_q:
        :param h: <I dont know why this is here - I have set it to None>
        :param callback:
        :param args:
        :param kwargs:
        :return:
        """
        if not self.has_dispatched:
            self.ctxs = ctxs
            self.cmd_qs = cmd_qs
            self.build_kernels(gpus,cpus,ctxs)
            self.has_dispatched = True

        else:
            ctxs = self.ctxs
            cmd_qs = self.cmd_qs

        start_time = time.time()
        partition = int(self.optm_device)




        device = self.device
        dag = self.task_dag_object


        if self.is_finished():
            dag.finished_tasks.add(self)

            assert len(self.occupied_devices) == 0, "occupied devices :-\n"+self.occupied_devices
            assert len(self.free_devices) == len(self.allocated_devices),"free_devices :- \n"+str(self.free_devices)+"\n allocate_devices :- \n"+str(self.allocated_devices)

            rqLock.acquire()
            logging.debug("Task on device " + str(self.device) + " - "+ str(task.id) + "finished " + \
            " free devices ",self.free_devices)

            #print "ready queue", ready_queue[self.device]
            for dev_number in self.free_devices:
                ready_queue[self.device].append(dev_number)

            print "Task" , str(self.id) ," finished " ," State of ready queue :- ", ready_queue

            if device == 'gpu':
                global nGPU
                nGPU += 1
            else:
                global nCPU
                nCPU += 1

            rqLock.notifyAll()
            rqLock.release()
            logging.debug("Task "+self.id+" finished")
            return

        self.free_kernels_lock.acquire()
        # while not self.free_kernels:
        #     self.free_kernels_lock.wait()

        kernel = self.free_kernels.pop(0)
        self.free_kernels_lock.release()

        # done_events = []
        # if not self.free_kernels:
        #     self.refresh_free_kernels(dag)
        # kernel = self.get_some_free_kernel()
        #if(kernel):
            #print "kernel dispatching  dev_no" , kernel , dev_type, dev_no

        # if kernel in self.processed_kernels:
        #     raise Exception('Kernel {0}_{1} has already been processed.'.format(kernel.name, kernel.id))

        # self.processed_kernels.add(kernel)
        # if not self.get_kernel_parents(kernel.id, dag) <= set(self.finished_kernels):
        #     raise Exception('Kernel {0}_{1} has unmet Dependencies'.format(kernel.name, kernel.id))
        self.prepare_kernel(kernel.id)
        kernel.random_data()
        #print "data from the kernel " , kernel.data
        self.load_dependent_data_and_buffers(kernel)
        #print "after " , kernel.data ,"dep ", dag.get_kernel_parent_ids(kernel.id)
        # if( kernel.task_dag_object.flag == 0):
        #     kernel.task_dag_object.start_time = time.time()
        #     kernel.task_dag_object.flag = 1
        logging.debug("dispatch_single :- choosing which CommandQueue to dispatch in ")

        logging.debug("dispatch single :- acquiring device lock")
        self.devices_lock.acquire()
        while not (len(self.free_devices) > 0):
            self.devices_lock.wait()
        device = self.free_devices.pop(0)
        self.occupied_devices.append(device)
        self.devices_lock.release()
        logging.debug("dispatch_single :- releasing device lock")

        kernel.dag_device = device
        logging.debug("dispatch_single :- CommandQueue number "+str(device))

        if self.device == "gpu":
            gpu = device
            cpu = -1
            partition = 10

        else:
            cpu = device
            gpu = -1
            partition = 0

        _,s, d = kernel.dispatch_dag(dag ,h ,  gpu, cpu, ctxs, cmd_qs, partition=partition, callback=callback, *args, **kwargs )
        done_events.extend(d)

        if self.is_processed():
            dag.processed_tasks.add(self)

        return done_events , kernel


    def dispatch_all(self, cmd_qs = None,h=None, callback=blank_fn, *args, **kwargs):
        """
        Dispatch any of the free kernel of the task on the particular device
        :param gpu:
        :param cpu:
        :param ctxs:
        :param cmd_q:
        :param callback:
        :param args:
        :param kwargs:
        :return:
        """
        dag = self.task_dag_object
        if not self.has_dispatched:
            self.has_dispatched = True


        ctxs = dag.pointers["ctxs"]
        cmd_qs = dag.pointers["cmd_qs"]

        start_time = time.time()
        partition = int(self.optm_device)

       

        device = self.device




        self.resource_lock.acquire()

        while (((len(self.free_devices) == 0) or (len(self.free_kernels) == 0)) and (not self.is_finished())):
            self.resource_lock.wait()
            # logging.debug("On device "+str(self.device)+ " and waiting" )
            #print(self.free_kernels,self.free_devices,self.is_finished())

        if not self.free_kernels:
            #logging.debug("On device "+str(self.device)+" and leaving without dispatching anything :( ")
            pass

        free_kernel_ids = [k.id for k in self.free_kernels]
        logging.info("EVENT: Free kernels: " + str(free_kernel_ids) +"for task " + str(self.id))



        dependencies = self.dependencies
        user_deps = []

        kernels_that_will_be_dispatched = []

        temp=time.time()

        num_dispatched = 0


        while self.free_kernels: #maximal matching of free kernels and free devices
            device = self.free_devices.pop(0)   #device to be allocated to the kernel
            self.occupied_devices.append(device)
            kernel = self.free_kernels.pop(0)   #the kernel which will be allocated to the device

            kernels_that_will_be_dispatched.append(kernel)



            logging.info("Kernel "+str(kernel.id)+" assigned device number "+str(device))

            # if kernel in self.processed_kernels:
            #     raise Exception('Kernel {0}_{1} has already been processed.'.\
            #     format(kernel.name, kernel.id))

            # self.processed_kernels.add(kernel)
            self.prepare_kernel(kernel.id)
            kernel.random_data()
            self.load_dependent_data_and_buffers(kernel)
            kernel.dag_device = device



            if self.device == "gpu":
                gpu = device  ##this is where the supposed device is assigned
                cpu = -1
                partition = 10

            else:
                cpu = device
                gpu = -1
                partition = 0

            logging.info("DISPATCH ALL :"+self.device+" " + str(kernel.id) + " " + str(kernel.name))
            
            ##chkz
            if kernel.id not in dependencies:
                exec_dep = None
            else:
                exec_dep = dependencies[kernel.id]

            succesors = dag.get_kernel_children_ids(kernel.id)
            read_cb = True
            if len(succesors) == 0:
                read_cb = False

            #read_cb=True

            logging.info("EVENT: Dispatching" + str(kernel.id) + " to device" + str(cpu) + "," + str(gpu)+" with rb set to " +str(read_cb)) 
            # if kernel.id in dependencies:
            #     dependent_ids = [k.id for k in dependencies[kernel.id]]
            #     logging.info("DEPENDENCIES " + str(kernel.id) + str(dependent_ids))

            d,user_dep = kernel.dispatch_dag(dag ,h ,  gpu, cpu, ctxs,\
             cmd_qs, partition=partition, callback=callback,n_chunks = just_for_testing_num_chunks,
             exec_dep = exec_dep, read_cb = read_cb)

            user_deps.append(user_dep)

            # kernel.host_events[0].write_start = time.time()
            # user_dep[self.device].set_status(cl.command_execution_status.COMPLETE)



            nd_range_event = kernel.nd_range_event

            self.add_finished_kernels([kernel])


            for child in succesors:
                child_kernel = dag.kernels[child]
                child_task = child_kernel.task_object
                if not (child_task == self):
                    continue
                # if child in self.processed_kernels:
                #     continue

                dag.kernel_flags[child].finished_parents +=1
                dependencies[child] = nd_range_event
                logging.info("EVENT: number of finished parents for child " + str(child) + " " + str(dag.kernel_flags[child].finished_parents))
                if dag.kernel_flags[child].is_ready():
                    self.add_free_kernels([child_kernel])

            self.occupied_devices.remove(device)
            self.free_devices.append(device)
            done_events.extend(d)




        if self.is_processed():
            dag.processed_tasks.add(self)

        st = time.time()
        for kernel in kernels_that_will_be_dispatched:
            kernel.host_events[0].write_end = st

        # logging.critical("TIME to Enqueue everything - {}".format(time.time()-temp))
        #testing uncomment these two loops later

        for dep in user_deps:
            for key in ["gpu","cpu"]:
                dep[key].set_status(cl.command_execution_status.COMPLETE)


        # for device_id in self.free_devices:
        #     cmd_qs[self.device][device_id].flush()







        self.resource_lock.notifyAll()
        self.resource_lock.release()




        if self.is_finished():
            dag.finished_tasks.add(self)



            frontier_Q_lock.acquire()
            frontier_Q_lock.notifyAll()
            frontier_Q_lock.release()

            assert len(self.occupied_devices) == 0, "occupied devices :-\n"+self.occupied_devices
            assert len(self.free_devices) == len(self.allocated_devices),"free_devices :- \n"+str(self.free_devices)+"\n allocate_devices :- \n"+str(self.allocated_devices)

            rqLock.acquire()


            for dev_number in self.free_devices:
                ready_queue[self.device].append(dev_number)

            logging.info("Task" + str(self.id) +" finished on device " + str(self.device) + " State of ready queue :- " + str(ready_queue))

            if device == 'gpu':
                global nGPU
                nGPU += 1
                global est_gpu
                est_gpu = 0.0
            else:
                global nCPU
                nCPU += 1
                global est_cpu
                est_cpu = 0.0

            rqLock.notifyAll()
            rqLock.release()



        return done_events


    def remove_kernel(self, kernel):
        """
        Removes given kernel from this task.
        """
        import operator
        if kernel in self.kernels:
            self.kernels.remove(kernel)
            self.modify_device_requirement(set(kernel), operator.isub)
        else:
            raise Exception("Given kernel is not a subset of this task")

    def add_kernels_from_task(self, task):
        """
        Merges a child task into itself.
        """
        self.kernels.update(task.get_kernels())
        self.kernel_ids.update(task.get_kernel_ids())
        # self.modify_device_requirement(task.get_kernels(), operator.iadd)

    def modify_device_requirement(self, kernels, op=operator.iadd):
        for k in kernels:
            dev_req = k.get_device_requirement()
            for key in self.dev_requirement:
                self.dev_requirement[key] = op(self.dev_requirement[key], dev_req.get(key, 0))

    def get_device_requirement(self):
        return self.dev_requirement

    def get_first_kernel(self):
        return list(self.kernels)[0]

    def get_kernels(self):
        return self.kernels

    def get_kernel_names(self):
        """
        Returns list of names pertaining to each kernel 
        :return:
        :rtype:
        """
        return map(lambda k: k.name, self.get_kernels())
    
    def get_kernel_ids(self):
        return self.kernel_ids

    def get_kernels_sorted(self, dag):
        import networkx as nx
        return map(lambda kid: dag.get_kernel(kid),
                   nx.algorithms.topological_sort(dag.get_skeleton_subgraph(map(lambda k: k.id, self.get_kernels()))))

    def get_kernel_ids_sorted(self, dag):
        return map(lambda k: k.id, self.get_kernels_sorted(dag))

    def is_supertask(self):
        return len(self.get_kernels()) > 1

    def random_data(self):
        raise NotImplementedError

    def is_finished(self):
        return len(self.kernels) == len(self.finished_kernels)

    def is_processed(self):
        return len(self.kernels) == len(self.processed_kernels)



    def get_kernel_parents(self, kernel_id, dag):
        """
        Returns set of Kernel objects
        :param kernel_id:
        :param dag:
        :return:
        """
        return set(map(lambda k: dag.get_kernel(k), dag.get_kernel_parent_ids(kernel_id))) & self.get_kernels()

    def get_kernel_children(self, kernel_id, dag):
        return set(dag.get_kernel_children_ids(kernel_id)) & self.get_kernels()




class TaskDAG(object):
    def __init__(self,kernel_dag,gpus,cpus,ctxs,cmd_qs,dag_number,dataset=1024,map_all=False,all_map_to_gpu = True,use_predefined_mapping=False,ex_stats_file=None,cluster_scheduler=False, heft_scheduler=False):
        import networkx as nx
        logging.debug("pyschedcl.TaskDAG : number of dag nodes : " + str(len(kernel_dag)))
        self.id = dag_number
        self.kernels = dict()
        self.skeleton = nx.DiGraph()
        
        self.pointers = {"gpus":gpus,"cpus":cpus,"ctxs":ctxs,"cmd_qs":cmd_qs}

        self.kernel_parents = dict()
        self.kernel_flags = dict()
        self.free_kernels = []
        self.free_tasks = []



        self.task_to_kernels = defaultdict(list)
        self.id_to_task = dict()
        self.kernel_to_task = dict()


        self.finished_tasks = set()
        self.processed_tasks = set()

        for src in kernel_dag:
            assert 'id' in src #each kernel is the json file should have an id
            kernel = Kernel(src, self, src["globalWorkSize"][0], src['partition'])

            #print "evaluating arguments"
            #kernel.eval_vargs()
            #print kernel.variable_args
            #time.sleep(1)
            kernel.optm_device = src['partition']


            # with open(SOURCE_DIR+"database/times"+str(src["globalWorkSize"][0]) +".txt") as f1:
            #     for line in f1:
            #         if(line != '\n'):
            #             line1 = line.split(" ")[0]
            #             if(line1.split('/')[1].split('.')[0] == kernel.name):
            #                 kernel.optm_device = line.split(' ')[4]

            if use_predefined_mapping:
                if map_all:
                    src['task'] = 0
                    if all_map_to_gpu:
                        src['partition'] = 10
                        kernel.optm_device = 10
                    else:
                        src['partition'] = 0
                        kernel.optm_device = 0


                if 'task' in src:
                    ###CURRENTLY NOT SUPPORTED###
                    self.predefined_task_mappings = True
                    self.task_to_kernels[src['task']].append(kernel.id)
                    logging.debug("Using predefined task mapping")


                else:
                    self.predefined_task_mappings = False    #?? should be false??
                    self.task_to_kernels[kernel.id].append(kernel.id)

            else:
                self.predefined_task_mappings = False    #?? should be false??
                self.task_to_kernels[kernel.id].append(kernel.id)
            
            self.kernels[kernel.id] = kernel
            self.skeleton.add_node(src['id'], req=kernel.get_device_requirement())

            if 'depends' in src :
                if type(src['depends']) == list and len(src['depends']) > 0:
                    self.kernel_parents[src['id']] = src['depends']
                    for i in src['depends']:
                        self.skeleton.add_edge(i, src['id'])
                    self.kernel_flags[src['id']] = kernelFlags(parents = self.kernel_parents[src['id']])
            else:
                self.kernel_flags[src['id']] = kernelFlags(parents = [])

        # Setting up optimal time information

        

        for kernel_id,flags in self.kernel_flags.items():
            logging.debug(str(kernel_id)+" 's ready status "+str(flags.is_ready()))

        for node in self.skeleton.nodes():
            if node not in self.kernel_parents:
                self.free_kernels.append(node)

        for task_id,kernels in self.task_to_kernels.items():
            new_task = Task(self.get_kernel(kernels[0]),self)
            self.id_to_task[task_id] = new_task
            self.kernel_to_task[kernels[0]] = new_task
            for kernel in kernels[1:]:
                new_task.add_kernel(self.get_kernel(kernel))
                self.kernel_to_task[kernel] = new_task

        logging.debug("pyschedcl.init : initial_free_kernels :"+ str(self.free_kernels))

        # Create task G

        mapping = lambda s: self.kernel_to_task[s] #each node as a task object
        self.G = nx.relabel_nodes(self.skeleton, mapping, copy=True)  #relabeling nodes to their corresponding task objects

        if ex_stats_file:
            self.set_optimal_ex_time_information(ex_stats_file)

        # Obtain task clusters if cluster_scheduler is set to True
        if cluster_scheduler:
            self.compute_blevel_ranks()
            task_device_bias_map = self.obtain_task_clusters()
            kernel_list_cpu = [] 
            kernel_list_gpu = [] 
            for kernel,device_bias in task_device_bias_map.items():
                logging.info(str(kernel) + " " + str(device_bias))
                if device_bias == 'cpu':
                    kernel_list_cpu.append(self.kernels[kernel].task_object)
                else:
                    kernel_list_gpu.append(self.kernels[kernel].task_object)

            self.merge_task_list(kernel_list_cpu, 0)
            self.merge_task_list(kernel_list_gpu, 10)
        elif heft_scheduler:
            self.compute_blevel_ranks()
            task_device_bias_map = self.run_static_heft()
            kernel_list_cpu = [] 
            kernel_list_gpu = [] 
            for kernel,device_bias in task_device_bias_map.items():
                logging.info(str(kernel) + " " + str(device_bias))
                if device_bias == 'cpu':
                    kernel_list_cpu.append(self.kernels[kernel].task_object)
                else:
                    kernel_list_gpu.append(self.kernels[kernel].task_object)

            self.merge_task_list(kernel_list_cpu, 0)
            self.merge_task_list(kernel_list_gpu, 10)

        logging.info("id to tasks map: "+ str(self.id_to_task))
        logging.info("kernel to task map " + str(self.kernel_to_task))

        #Set-free-kernels-for-each-task

        for kernel_id in self.free_kernels:
            self.kernel_to_task[kernel_id].add_free_kernels([self.kernels[kernel_id]])


        for _,task in self.id_to_task.items():
            task.build_kernels(gpus, cpus, ctxs)
            if task.free_kernels:
                self.free_tasks.append(task)


        for task in self.free_tasks:
            logging.debug("pyschedcl.taskDAG.init : printing free tasks :- "+\
                str(task.id)+ " and its free kernels "+ str([k.id for k in task.free_kernels]))

        self.ex_map = None

        
        

    def set_optimal_ex_time_information(self,ex_stats_file):
        import json
        with open(ex_stats_file,'r') as g:
            self.ex_map=json.loads(g.read())
        
        for node in self.skeleton.nodes():
            u = self.kernels[node]
            u.exec_time['cpu']=convert_to_milliseconds(float(self.ex_map[u.name]['exec_cpu']))
            u.exec_time['gpu']=convert_to_milliseconds(float(self.ex_map[u.name]['exec_gpu']))
            u.exec_time['h2d_time']=convert_to_milliseconds(float(self.ex_map[u.name]['h2d_time']))
            u.exec_time['d2h_time']=convert_to_milliseconds(float(self.ex_map[u.name]['d2h_time']))
            if 'gpu_delay' in self.ex_map[u.name].keys():
                u.exec_time['gpu_delay']=convert_to_milliseconds(float(self.ex_map[u.name]['gpu_delay']))
            if 'gpu_delay' in self.ex_map[u.name].keys():
                u.exec_time['cpu_delay']=convert_to_milliseconds(float(self.ex_map[u.name]['cpu_delay']))
            if u.exec_time['cpu'] > u.exec_time['gpu']:
                u.optm_device=10
                u.task_object.set_device(10)
            else:
                u.optm_device=0
                u.task_object.set_device(0)

        
        for edge in self.skeleton.edges():
            
            u, v = edge
            k_source = self.kernels[u].name
            k_target = self.kernels[v].name
            size_input_buf = convert_to_milliseconds(float(self.ex_map[k_target]['h2d_bytes']))
            size_output_buf = convert_to_milliseconds(float(self.ex_map[k_source]['d2h_bytes']))
            time_input_buf = convert_to_milliseconds(float(self.ex_map[k_target]['h2d_time']))
            time_output_buf = convert_to_milliseconds(float(self.ex_map[k_source]['d2h_time']))
            self.skeleton[u][v]['weight'] = min(size_output_buf, size_input_buf)
            self.skeleton[u][v]['time'] = min(time_output_buf, time_input_buf)

    def print_dag_info(self):
        for i in self.skeleton.nodes():
            print "Kernel", self.kernels[i].id 

    def get_kernel(self,kid):
        return self.kernels[kid]

    def get_kernel_parent_ids(self,kid):
        return self.skeleton.predecessors(kid)

    def get_kernel_children_ids(self,kid):
        return self.skeleton.successors(kid)

    def get_kernel_children(self, kernel):
        kernel_successors = []
        kernel_successor_ids = self.get_kernel_children_ids(kernel.id)
        for k in kernel_successor_ids:
            kernel_successors.append(self.kernels[k])
        return kernel_successors

    def get_kernel_parents(self, kernel):
        kernel_predecessors = []
        kernel_predecessor_ids = self.get_kernel_parent_ids(kernel.id)
        for k in kernel_predecessor_ids:
            kernel_predecessors.append(self.kernels[k])
        return kernel_predecessors
        
        
    def get_cpu_time(self,node):
        k = self.kernels[node]
        return k.exec_time['cpu']

    def get_gpu_time(self,node):
        k = self.kernels[node]
        return k.exec_time['gpu']

    def get_h2d_time(self,node):
        k = self.kernels[node]
        return k.exec_time['h2d_time']

    def get_d2h_time(self,node):
        k = self.kernels[node]
        return k.exec_time['d2h_time']

    
    def get_device_preference(self,node,device_bias):
        predecessors = self.get_kernel_parent_ids(node)
        gpu_time = self.kernels[node].exec_time['gpu']
        cpu_time = self.kernels[node].exec_time['cpu']
        for p in predecessors:
            if device_bias[p]=="gpu":
                gpu_time += self.kernels[node].exec_time['h2d_time']-self.kernels[p].exec_time['d2h_time']
            else:
                cpu_time += self.kernels[node].exec_time['h2d_time']
        if gpu_time > cpu_time:
            return "cpu"
        else:
            return "gpu"

    def set_device_preference_of_multiple_kernels(self,gpu_queue,cpu_queue):
        import heapq
        cpu_time = gpu_time = 0.0
        cpu_time = sum([x for _,x in cpu_queue])
        gpu_time = sum([x for _,x in gpu_queue])
        total_time = max(cpu_time, gpu_time)
        print cpu_queue,gpu_queue,cpu_time,gpu_time
        if gpu_time > cpu_time:
            while True:
                if not gpu_queue:
                    break
                node,gpu_min_time = heapq.heappop(gpu_queue)
                cpu_time = self.get_cpu_time(node)
                curr_total_time = max(gpu_time-gpu_min_time,cpu_time+self.get_h2d_time(node))
                if curr_total_time < total_time:
                    heapq.heappush(cpu_queue, (node,cpu_time))

                else:
                    heapq.heappush(gpu_queue, (node,gpu_min_time))                    
                    break
        else:
            while True:
                if not cpu_queue:
                    break
                node,cpu_min_time = heapq.heappop(cpu_queue)
                gpu_time = self.get_gpu_time(node)
                curr_total_time = max(cpu_time-cpu_min_time,gpu_time+self.get_h2d_time(node))
                if curr_total_time < total_time:
                    heapq.heappush(gpu_queue, (node,gpu_time))

                else:
                    heapq.heappush(cpu_queue, (node,cpu_min_time))                    
                    break




    def finished(self):
        return len(self.finished_tasks) == len(self.id_to_task)

    def get_tasks(self):
        return self.G.nodes()

    def get_tasks_sorted(self):

         import networkx as nx

         return nx.algorithms.topological_sort(self.G)


    def get_all_task_dependencies(self):
        return self.G.edges()

    def get_task_parents(self, task):
        return self.G.predecessors(task)

    def get_task_children(self, task):
        return self.G.successors(task)

    def get_task_children_kernel_ids(self, task):
        return map(lambda t: t.get_kernel_ids(), self.G.successors(task))


    def print_information(self):

        print self.skeleton.nodes()
        print self.skeleton.edges()
        for t in self.kernels:
            print str(t) + " " + self.kernels[t].name
        for node in self.G.nodes():
            print node.get_kernel_ids(), node.get_kernel_names()
            # print node.rank_values[rank_name]
            # print "Ex Time: " + str(node.projected_ex_time)
        for edge in self.G.edges():
            u, v = edge
            print u.get_kernel_ids(),
            print "--->",
            print v.get_kernel_ids()

    def print_task_information(self):

        counter = 0
        for node in self.G.nodes():
            print counter, node.get_kernel_ids(), node.get_kernel_names()
            counter +=1
            # print node.rank_values[rank_name]
            # print "Ex Time: " + str(node.projected_ex_time)
        for edge in self.G.edges():
            u, v = edge
            print u.get_kernel_ids(),
            print "--->",
            print v.get_kernel_ids()

    def print_kernel_information(self):
        for node in self.skeleton.nodes():
            print node, self.kernels[node].name,self.kernels[node].exec_time
        for edge in self.skeleton.edges():
            u,v =edge
            print u,v,self.skeleton[u][v]['weight'],"bytes",self.skeleton[u][v]['time'],"seconds"

    def merge_tasks(self, t1, t2, device_bias):
        """
        :param t1:
        :type t1: Task
        :param t2:
        :type t2: Task
        :return:
        """
        # dependencies = set().union(*[set(self.get_kernel_parent_ids(kid)) for kid in t2.get_kernel_ids()])
        # print "Sets for checking dependenciers",set(t1.get_kernel_ids()), dependencies
        # print "adding kernels ", t2.get_kernel_ids(),"to ",t1.get_kernel_ids()
        if set(t1.get_kernel_ids()) >= 0:
            t1.add_kernels_from_task(t2)
        else:
            raise Exception('Some dependent kernels are not part of this task.')

        for kid in t2.get_kernel_ids():
            self.kernel_to_task[kid] = t1

        for k in t2.kernels:
            k.task_object = t1
            k.optm_device = device_bias
        # t1.rank = t2.rank + t1.rank
        # t1.optm_device = (len(list(t1.kernels))*int(t1.optm_device) + int(t2.optm_device))/(len(list(t1.kernels)) + 2)
        self.update_dependencies(t1)
        self.G.remove_node(t2)
        self.id_to_task.pop(t2.id)

        t1.set_partition(device_bias)
        # print "After merging task contatins: ",t1.get_kernel_ids()

    def merge_task_list(self, t,device_bias):
        
        t1 = t[0]
        for t2 in t[1:]:
            self.merge_tasks(t1, t2, device_bias)

    def update_dependencies(self, task):
        """
        Updates task dependencies. Call this whenever a task is modified. Adds or remove edges to task dag based on
        skeleton kernel dag for the given task.
        :param task:
        :return:
        """
        p, c = set(self.get_task_parents(task)), set(self.get_task_children(task))
        pt, ct = set(), set()
        for kid in task.get_kernel_ids():
            for pkid in self.get_kernel_parent_ids(kid):
                pt.add(self.kernel_to_task[pkid])
            for ckid in self.get_kernel_children_ids(kid):
                ct.add(self.kernel_to_task[ckid])
        pt -= set([task])
        ct -= set([task])
        for t in pt - p:
            self.G.add_edge(t, task)
        for t in ct - c:
            self.G.add_edge(task, t)
        for t in p - pt:
            self.G.remove_edge(t, task)
        for t in c - ct:
            self.G.remove_edge(task, t)

    def dump_graph(self, file_name,rank=False,timestamps=None, print_all_times=False):
        '''
        G = dag.skeleton
        plot = figure(title="Graph Plotting", x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
                      tools="", toolbar_location=None)
        graph = from_networkx(G, nx.spring_layout, scale=2, center=(0, 0))
        plot.renderers.append(graph)
        output_file("visualize_graph.html")
        show(plot)
        '''
        import pygraphviz as pgv
        import random
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        S = self.skeleton
        T = self.G

        for node in S.nodes():
            str_info = str(self.kernels[node].id)+":"+self.kernels[node].name
            if rank:
                str_info += "\n" + str(self.kernels[node].rank)
            if timestamps:
                import json
                relative_timestamps,total_time = adjust_zero(timestamps)  
                
                kernel = self.kernels[node]
                kernel_timestamps = relative_timestamps[kernel.name+str(kernel.id)]
                start_time = kernel_timestamps["nd_range"]["device_start"] 
                end_time = kernel_timestamps["nd_range"]["device_end"] 
                str_info +=  " " + str(start_time) + " " + str(end_time)
            if print_all_times:
                kernel = self.kernels[node]
                str_info += " gpu: " + str(kernel.exec_time['gpu']) + " cpu: " + str(kernel.exec_time['cpu']) + " h2d: " + str(kernel.exec_time['h2d_time']) + " d2h: " + str(kernel.exec_time['d2h_time'])
            G.add_node(node, label=str_info)
        for edge in S.edges():
            u, v = edge
            G.add_edge(u, v)
        for node in T:
            if node.is_supertask():
                r = lambda: random.randint(0, 255)
                # print('#%02X%02X%02X' % (r(),r(),r()))
                if node.device == "gpu":
                    color = "#a4e1a2"
                else:
                    color = "#93b6ff"
                kid = node.get_kernel_ids()

                for k in kid:
                    n = G.get_node(k)
                    n.attr['shape'] = 'square'
                    n.attr['fillcolor'] = color
                    # n.attr['label'] = str_info
        G.layout(prog='dot')
        # file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(file_name)


    def dump_device_biases(self, file_name,task_device_bias_map):
        '''
        
        '''
        import pygraphviz as pgv
        G = pgv.AGraph(strict=False, directed=True)
        G.node_attr['style'] = 'filled'
        S = self.skeleton
        

        for node in S.nodes():
            str_info = str(self.kernels[node].id)+":"+self.kernels[node].name
            G.add_node(node, label=str_info)
            n = G.get_node(node)
            if task_device_bias_map[node]=='gpu':           
                n.attr['fillcolor'] = '#a4e1a2'
            else:
                n.attr['fillcolor'] = '#93b6ff'

        for edge in S.edges():
            u, v = edge
            G.add_edge(u, v)
                
        G.layout(prog='dot')
        # file_name = "dag_" + str(dag.dag_id) + "_contracted.png"
        G.draw(file_name)

    def compute_blevel_ranks(self):
        import networkx as nx
        rev_top_list = list(nx.topological_sort(self.skeleton))
        rev_top_list.reverse()
        for node in rev_top_list:
            if self.skeleton.out_degree(node) == 0:
                kernel = self.kernels[node]
                kernel.rank=(kernel.exec_time['cpu']+kernel.exec_time['gpu']+kernel.exec_time['d2h_time']+kernel.exec_time['h2d_time'])/2
        for node in rev_top_list:
            max_rank = 0.0
            kernel = self.kernels[node]
            # print "ECOn of ",node, task.ECO
            kernel.rank=(kernel.exec_time['cpu']+kernel.exec_time['gpu']+kernel.exec_time['d2h_time']+kernel.exec_time['h2d_time'])/2
            max_rank = kernel.rank
            for child in self.get_kernel_children(kernel):
                if child.rank + kernel.rank > max_rank:
                    max_rank = child.rank + kernel.rank 
            kernel.rank = max_rank
            kernel.task_object.rank = kernel.rank

    def compute_tlevel_ranks(self):
        import networkx as nx
        rev_top_list = list(nx.topological_sort(self.skeleton))
        # rev_top_list.reverse()
        for node in rev_top_list:
            if self.skeleton.in_degree(node) == 0:
                kernel = self.kernels[node]
                kernel.rank=kernel.exec_time['cpu']+kernel.exec_time['gpu']+kernel.exec_time['h2d_time']
        for node in rev_top_list:
            max_rank = 0.0
            kernel = self.kernels[node]
            # print "ECOn of ",node, task.ECO
            kernel.rank=kernel.exec_time['cpu']+kernel.exec_time['gpu']+kernel.exec_time['h2d_time']
            max_rank = kernel.rank
            for child in self.get_kernel_parents(kernel):
                if child.rank + kernel.rank > max_rank:
                    max_rank = child.rank + kernel.rank 
            kernel.rank = max_rank
            kernel.task_object.rank = kernel.rank

    def run_static_heft(self):
        import networkx as nx
        top_list = list(nx.topological_sort(self.skeleton))
        import heapq
        ranked_queue = []
        for node in top_list:
            heapq.heappush(ranked_queue,(node, -self.kernels[node].rank))
        device_bias = {} 
        est = {'cpu': 0.0, 'gpu': 0.0}
        while len(ranked_queue)!=0:
            kernel,rank = heapq.heappop(ranked_queue)
            exec_cpu = self.kernels[kernel].exec_time['cpu']
            exec_gpu = self.kernels[kernel].exec_time['gpu'] + self.kernels[kernel].exec_time['h2d_time'] + self.kernels[kernel].exec_time['d2h_time']
            if est['cpu'] + exec_cpu < est['gpu'] + exec_gpu:
                device_bias[kernel]='cpu'
                est['cpu'] += exec_cpu
            else:
                device_bias[kernel]='gpu'
                est['gpu'] += exec_cpu

        return device_bias


    def make_levels(self):
        import networkx as nx
        node_level = dict()
        levels = defaultdict(list)
        
        for node in nx.algorithms.topological_sort(self.skeleton):
            pred = self.skeleton.predecessors(node)
            if not pred:
                print "node value",node
                node_level[node] = 0
            else:
                node_level[node] = max(map(lambda x: node_level[x], pred)) + 1
            levels[node_level[node]].append(node)
        return map(lambda x: x[1], sorted(levels.items(), key=lambda x: x[0]))


       


    def obtain_task_clusters(self):
        import heapq
        class kernel_priority(object):
            def __init__(self,node,blevel_rank,ex_time):
                self.node = node
                self.rank = blevel_rank
                self.ex_time = ex_time
            def __lt__(self,other):
                if self.ex_time == other.ex_time:
                    return self.rank < other.rank
                else:
                    return self.ex_time < other.ex_time
            def get_node_and_time(self):
                return self.node,self.ex_time
            def get_info(self):
                return self.node,self.ex_time,self.rank

        def get_task_device_times(level,task_device_times_map,task_device_bias_map):
            for node in level:
                predecessors = self.get_kernel_parent_ids(node)
                gpu_time = self.kernels[node].exec_time['gpu'] + self.kernels[node].exec_time['h2d_time']
                cpu_time = self.kernels[node].exec_time['cpu']
           
                for p in predecessors:
                    if task_device_bias_map[p]=="cpu":
                        gpu_time += self.skeleton[p][node]['time']
                    else:
                        cpu_time += self.skeleton[p][node]['time']
                task_device_times_map[node] = (cpu_time, gpu_time)



        def set_task_device_bias(level,task_device_times_map,task_device_bias_map):
            cpu_queue = []
            gpu_queue = []
            
            for node in level:
                cpu_time, gpu_time = task_device_times_map[node]
                # heapq.heappush(gpu_queue,(node, gpu_time))
                # heapq.heappush(cpu_queue,(node, cpu_time)) 
                heapq.heappush(gpu_queue,kernel_priority(node, -self.kernels[node].rank,gpu_time))
                heapq.heappush(cpu_queue,kernel_priority(node, -self.kernels[node].rank,cpu_time))
                
            from copy import deepcopy
            tasks = deepcopy(level)
            total_cpu_time = total_gpu_time = 0
            while tasks:
                # t1,_ = heapq.heappop(cpu_queue)
                # t2,_ = heapq.heappop(gpu_queue)

                t1,t1_time,t1_rank = (heapq.heappop(cpu_queue)).get_info()
                t2,t2_time,t2_rank = (heapq.heappop(gpu_queue)).get_info()
                if t1 == t2:
                    task = t1
                    cpu_time,gpu_time = task_device_times_map[task]
                    if total_cpu_time + cpu_time > total_gpu_time + gpu_time: #GPU Timing is better
                        task_device_bias_map[task]='gpu'
                        total_gpu_time += gpu_time
                    else:
                        task_device_bias_map[task]='cpu'
                        total_cpu_time += cpu_time
                    tasks.remove(task)
                    logging.info("ALGORITHM: 0. Selecting task "+str(task) + " with rank" + str(t1_rank) + "and mapping to " + task_device_bias_map[task])
                else:
                    # Check for four cases
                    t1_cpu,t1_gpu = task_device_times_map[t1]
                    t2_cpu,t2_gpu = task_device_times_map[t2]
                    timing_t1_cpu_t2_gpu = max(total_cpu_time+t1_cpu,total_gpu_time+t2_gpu)
                    timing_t1_gpu_t2_gpu = max(total_cpu_time,total_gpu_time+t1_gpu,t2_gpu)
                    timing_t1_cpu_t2_cpu = max(total_cpu_time+t1_cpu+t2_cpu,total_gpu_time)
                    timings = [timing_t1_cpu_t2_gpu,timing_t1_gpu_t2_gpu,timing_t1_cpu_t2_cpu]

                    min_index = timings.index(min(timings))
                    if min_index == 0:
                        task_device_bias_map[t1]="cpu"
                        task_device_bias_map[t2]="gpu"
                        total_cpu_time += t1_cpu
                        total_gpu_time += t2_gpu


                    elif min_index == 1:
                        task_device_bias_map[t1]="gpu"
                        task_device_bias_map[t2]="gpu"
                        total_gpu_time += t1_gpu+t2_gpu
                    else:
                        task_device_bias_map[t1]="cpu"
                        task_device_bias_map[t2]="cpu"
                        total_cpu_time += t1_cpu+t2_cpu
                        logging.info("ALGORITHM: 1. Selecting task "+str(t1) + " with rank" + str(t1_rank)+"and mapping to " + task_device_bias_map[task])
                        logging.info("ALGORITHM: 2. Selecting task "+str(t2) + "with rank" + str(t2_rank)+"and mapping to " + task_device_bias_map[task])
                    tasks.remove(t1)
                    tasks.remove(t2)


        levels = self.make_levels()
        cpu_queue = []
        gpu_queue = []
        task_device_bias_map = {}
        task_device_times_map = {}
        get_task_device_times(levels[0],task_device_times_map,task_device_bias_map)
        print task_device_times_map
        for node in levels[0]:
            cpu_time, gpu_time = task_device_times_map[node]
            # heapq.heappush(gpu_queue,(node, gpu_time))
            # heapq.heappush(cpu_queue,(node, cpu_time))
            heapq.heappush(gpu_queue,kernel_priority(node, -self.kernels[node].rank,gpu_time))
            heapq.heappush(cpu_queue,kernel_priority(node, -self.kernels[node].rank,cpu_time)) 
        for l,level in enumerate(levels):
            total_gpu_time = 0.0
            total_cpu_time = 0.0
            if l == 0:
                for node in level:
                    
                    # node_gpu,gpu_time = heapq.heappop(gpu_queue)
                    # node_cpu,cpu_time = heapq.heappop(cpu_queue)
                    node_gpu,gpu_time,node_gpu_rank = (heapq.heappop(gpu_queue)).get_info()
                    node_cpu,cpu_time,node_cpu_rank = (heapq.heappop(cpu_queue)).get_info()
                    
                    if node_cpu == node_gpu:

                        if total_gpu_time + gpu_time> total_cpu_time + cpu_time:
                            task_device_bias_map[node_cpu]="cpu"
                            total_cpu_time +=cpu_time    
                            task_device_times_map[node_cpu]=cpu_time
                        else:
                            task_device_bias_map[node_gpu]="gpu"
                            task_device_times_map[node_gpu]=gpu_time
                            total_gpu_time +=gpu_time
                        
                    else:
                        
                        if gpu_time > cpu_time:
                            task_device_bias_map[node_cpu]="cpu"
                            task_device_times_map[node_cpu]=cpu_time
                            total_gpu_time += gpu_time    
                        else:

                            task_device_bias_map[node_gpu]="gpu"
                            task_device_times_map[node_gpu]=gpu_time
                            total_cpu_time += cpu_time    
            else:
                get_task_device_times(level,task_device_times_map,task_device_bias_map)
                set_task_device_bias(level,task_device_times_map,task_device_bias_map)

        return task_device_bias_map



    def allocate_tasks_to_device(self,levels):
        import heapq
        device_bias = dict()
        gpu_queue = []
        cpu_queue = []
        gpu_time = 0.0
        cpu_time = 0.0
        gpu_queue = []
        cpu_queue = []
        for node in levels[0]:
            heapq.heappush(gpu_queue,(node, self.get_gpu_time(node)))
            heapq.heappush(cpu_queue,(node, self.get_cpu_time(node)))
        for l,level in enumerate(levels):
            total_gpu_time = 0.0
            total_cpu_time = 0.0
            if l == 0:
                for node in level:
                    
                    node_gpu,gpu_time = heapq.heappop(gpu_queue)
                    node_cpu,cpu_time = heapq.heappop(cpu_queue)
                    
                    if node_cpu == node_gpu:

                        if total_gpu_time + gpu_time> total_cpu_time + cpu_time:
                            device_bias[node_cpu]="cpu"
                            total_cpu_time +=cpu_time    
                        else:
                            device_bias[node_gpu]="gpu"
                            total_gpu_time +=gpu_time
                        
                    else:
                        
                        if gpu_time > cpu_time:
                            device_bias[node_cpu]="cpu"
                            total_gpu_time += gpu_time    
                        else:
                            device_bias[node_gpu]="gpu"
                            total_cpu_time += cpu_time    

            else:
                for node in level:
                    print "getting device preference of", node

                    device_preference = self.get_device_preference(node, device_bias)
        
                    if device_preference == "gpu":
                        heapq.heappush(gpu_queue,(node, self.get_gpu_time(node)))
                    else:
                        heapq.heappush(cpu_queue, (node, self.get_cpu_time(node)))
                self.set_device_preference_of_multiple_kernels(cpu_queue,gpu_queue)
                while cpu_queue:
                    node,_ = heapq.heappop(cpu_queue)
                    device_bias[node]="cpu"
                while gpu_queue:
                    node,_ = heapq.heappop(gpu_queue)
                    device_bias[node]="gpu"
                       
                
        return device_bias








# class TaskDAG(object):
#     def __init__(self, srcs, dataset=1024):
#         import networkx as nx
#         self.kernels = dict()
#         self.tasks = dict()
#         self.unfinished_kernels = list()
#         self.skeleton = nx.DiGraph()
#         self.finished_kernels = set()
#         self.finished_tasks = set()
#         self.free_kernels = list()
#         self.free_tasks = list()
#         self.processing_tasks = list()
#         self.processed_tasks = set()
#         self.predefined_task_mappings = False
#         self.task_mappings = dict()
#         self.task_id_map = dict()
#         self.merged_task = list()
#         self.device_to_task = dict()
#         self.device_to_task['gpu'] = dict()
#         self.device_to_task['cpu'] = dict()
#         self.component = list()
#         self.free_tasks_merged = list()
#         self.unfinished_kernel_buffer = list()
#         self.depend_history = dict()
#         # following data type are used to store execution time
#         self.start_time = 0
#         self.end_time = 0
#         self.flag = 0
#         # end
#         for src in srcs:
#             if 'id' not in src: #kernel number
#                 raise
#             kernel = Kernel(src, self, src["globalWorkSize"][0], src['partition'])
#             lin1 = []
#             with open(SOURCE_DIR+"database/times"+str(src["globalWorkSize"][0]) +".txt") as f1:
#                 for line in f1:
#                     if(line != '\n'):
#                         line1 = line.split(" ")[0]
#                         if(line1.split('/')[1].split('.')[0] == kernel.name):
#                             kernel.optm_device = line.split(' ')[4]
#
#
#             ####TODO :- use defaultdicts
#             if 'task' in src:
#                 self.predefined_task_mappings = True
#                 try:
#                     self.task_mappings[src['task']].append(kernel.id)
#                 except KeyError:
#                     self.task_mappings[src['task']] = [kernel.id]
#             else:
#                 self.predefined_task_mappings = True   #?? should be false??
#                 try:
#                     self.task_mappings[kernel.id].append(kernel.id)
#                 except KeyError:
#                     self.task_mappings[kernel.id] = [kernel.id]
#
#             self.kernels[kernel.id] = kernel
#             self.skeleton.add_node(src['id'], req=kernel.get_device_requirement())
#             if 'depends' in src:
#                 self.depend_history[src['id']] = src['depends']
#                 for i in src['depends']:
#                     self.skeleton.add_edge(i, src['id'])
#         for node in self.skeleton.nodes():
#             if not self.get_kernel_parent_ids(node):
#                 self.free_kernels.append(node)
#
#         mapping = lambda s: Task(self.get_kernel(s))
#         self.G = nx.relabel_nodes(self.skeleton, mapping, copy=True)  #relabeling nodes to their corresponding task objects
#         for task in self.G.nodes():
#             for kid in task.get_kernel_ids():
#                 self.tasks[kid] = task
#
#         self.recently_added_kernels = self.free_kernels
#         self.recently_added_tasks = self.free_tasks
#         for task in self.G.nodes():
#             self.task_id_map[task.id] = task
#         if self.predefined_task_mappings:
#             for key in self.task_mappings.keys():
#                 if len(self.task_mappings[key]) > 1:
#                     t1 = self.tasks[self.task_mappings[key][0]]
#                     for t2 in self.task_mappings[key][1:]:
#                         self.merge_tasks(t1, self.tasks[t2])
#                         self.merged_task.append(t2)
#
#         for i in range(len(self.merged_task)):
#                 self.tasks.pop(self.merged_task[i])
#
#         count  = 1
#         new_task = dict()
#
#
#         for i in self.tasks:
#             new_task[count] = self.tasks[i]
#             count = count + 1
#
#         self.tasks = new_task
#
#         for j in self.task_mappings:
#                     self.tasks[j].update_task_info(self.task_mappings[j],self.free_kernels , self.kernels)
#
#         for j in self.tasks:
#             if(self.tasks[j].free_kernels):
#                 self.free_tasks.append(self.tasks[j])
#         rank_calculator(self)
#         blevel(self)
#         #self.dump_taskdag()
#
#     def dump_taskdag(self):
#         ans = []
#         dump_js = 0
#         for kernel in self.kernels.values():
#             dump_js = kernel.dump_jason()
#             print dump_js['id']
#             print list(self.depend_history.values())
#             if dump_js['id'] in list(self.depend_history.values()):
#                 dump_js['depends'] = self.depend_history[dump_js['id']]
#             ans.append(dump_js)
#
#         import json
#         with open('data/data.json', 'w') as outfile:
#             json.dump(ans, outfile)
#
#     def taskid(self , kernel_id):
#         for i in self.task_mappings:
#             for j in self.task_mappings[i]:
#                 if( j == kernel_id):
#                     return i
#         print "unknown kernnel_id, Dont know its task_id "
#         return 0
#
#     def update_dependencies(self, task):
#         """
#         Updates task dependencies. Call this whenever a task is modified. Adds or remove edges to task dag based on
#         skeleton kernel dag for the given task.
#         :param task:
#         :return:
#         """
#         p, c = set(self.get_task_parents(task)), set(self.get_task_children(task))
#         pt, ct = set(), set()
#         for kid in task.get_kernel_ids():
#             for pkid in self.get_kernel_parent_ids(kid):
#                 pt.add(self.tasks[pkid])
#             for ckid in self.get_kernel_children_ids(kid):
#                 ct.add(self.tasks[ckid])
#         pt -= set([task])
#         ct -= set([task])
#         for t in pt - p:
#             if(t not in c):
#                 self.G.add_edge(t, task)
#         for t in ct - c:
#             if( t not in p):
#                 self.G.add_edge(task, t)
#         for t in p - pt:
#             self.G.remove_edge(t, task)
#         for t in c - ct:
#             self.G.remove_edge(task, t)
#
#     def get_skeleton_subgraph(self, kernel_ids):
#         """
#         :param kernel_ids:
#         :type kernel_ids: list
#         :return: nx.Digraph
#         """
#         return self.skeleton.subgraph(kernel_ids)
#
#     def update_finished_kernels(self, kernel_id, *args, **kwargs):
#         self.finished_kernels.add(kernel_id)
#         successors = self.get_kernel_children_ids(kernel_id)
#         self.recently_added_kernels = []
#         for i in successors:
#             if set(self.get_kernel_parent_ids(i)) <= set(self.finished_kernels):
#                 self.recently_added_kernels.append(i)
#         self.free_kernels.extend(self.recently_added_kernels)
#         #self.tasks[ self.taskid(kernel_id) ].update_finished_kernels(self.get_kernel(kernel_id), self)
#         return self.recently_added_kernels, self.free_kernels
#
#     def get_finished_tasks(self):
#         return self.finished_tasks
#
#     def update_finished_tasks_1(self , task , kernel):
#
#         children = self.get_task_children(task)
#         self.recently_added_tasks = []
#         successors = self.get_kernel_children_ids(kernel.id)
#         finish_task = []
#         finish_task.extend(self.finished_tasks)
#         finish_task.append(task)
#         count = 0
#         for t in children:
#             for succ in successors:
#                 if self.kernels[succ] in  t.kernels:
#                         if set(self.get_task_parents(t)) <= set(finish_task):
#                             if t not in self.free_tasks:
#                                 self.recently_added_tasks.append(t)
#                                 count = count + 1
#         #print "start-update_finished_task" , count
#         self.free_tasks.extend(self.recently_added_tasks)
#         return self.recently_added_tasks, self.free_tasks
#
#     def update_finished_tasks(self, task):
#         try:
#             self.processing_tasks.remove(task)
#         except:
#             raise Exception('Task finished without processing?')
#         self.finished_tasks.add(task)
#         children = self.get_task_children(task)
#         self.recently_added_tasks = []
#         for t in children:
#             if set(self.get_task_parents(t)) <= set(self.get_finished_tasks()):
#                 if t not in self.free_tasks:
#                     self.recently_added_tasks.append(t)
#         self.free_tasks.extend(self.recently_added_tasks)
#         return self.recently_added_tasks, self.free_tasks
#
#     def kernel_data_transfer_size(self, kernel_r, kernel_s):
#         r = kernel_r.id
#         s = kernel_s.id
#         dt = 0
#         for key in ['input', 'io']:
#             for i in range(len(kernel_s.buffer_info[key])):
#                 if 'from' in kernel_s.buffer_info[key][i]:
#                     data_dep = kernel_s.buffer_info[key][i]['from']
#                     if data_dep['kernel'] is s:
#                         dt += kernel_r.get_data(data_dep['pos']).nbytes
#         return dt
#
#     def task_data_transfer_size(self, task_r, task_s):
#         tdt = 0
#         k_r = task_r.get_kernels()
#         k_s = task_s.get_kernels()
#         kernel_ids = map(lambda k: k.id, list(k_r) + list(k_s))
#         subgraph = self.get_skeleton_subgraph(kernel_ids)
#         for r, s in subgraph.edges():
#             tdt += self.kernel_data_transfer_size(self.kernels[r], self.kernels[s])
#         return tdt
#
#     def get_free_kernels(self):
#         """
#         Should return a list of kernels that don't have unmet dependencies.
#         """
#         return self.free_kernels
#
#     def get_kernel(self, kid):
#         """
#         Should return a kernel based on kernel id.
#         """
#         print self.kernels
#         print kid
#         return self.kernels.get(kid)
#
#     def get_kernel_parent_ids(self, kid):
#         """
#         Should return a list of kernel ids that are predecessors to given kernel.
#         """
#         return self.skeleton.predecessors(kid)
#
#     def get_kernel_children_ids(self, kid):
#         """
#         Should return a list of kernel ids that are successors to given kernel.
#         """
#         return self.skeleton.successors(kid)
#
#     def get_tasks(self):
#         return self.G.nodes()
#
#     def get_tasks_sorted(self):
#         import networkx as nx
#         return nx.algorithms.topological_sort(self.G)
#
#     def get_all_task_dependencies(self):
#         return self.G.edges()
#
#     def get_task_parents(self, task):
#         return self.G.predecessors(task)
#
#     def get_task_children(self, task):
#         return self.G.successors(task)
#
#     def get_free_tasks(self):
#         return self.free_tasks
#
#     def is_processed(self):
#         return set(self.G.nodes()) == self.processed_tasks
#
#     def process_free_task(self):
#         task = self.free_tasks.pop()
#         self.processing_tasks.append(task)
#         return task
#
#     def merge_tasks(self, t1, t2):
#         """
#         :param t1:
#         :type t1: Task
#         :param t2:
#         :type t2: Task
#         :return:
#         """
#         dependencies = set().union(*[set(self.get_kernel_parent_ids(kid)) for kid in t2.get_kernel_ids()])
#         if set(t1.get_kernel_ids()) >= 0:
#             t1.add_kernels_from_task(t2)
#         else:
#             raise Exception('Some dependent kernels are not part of this task.')
#         for kid in t2.get_kernel_ids():
#             self.tasks[kid] = t1
#
#         for k in t2.kernels:
#             k.task_object = t1
#         t1.rank = t2.rank + t1.rank
#         t1.optm_device = (len(list(t1.kernels))*int(t1.optm_device) + int(t2.optm_device))/(len(list(t1.kernels)) + 2)
#         self.update_dependencies(t1)
#         self.G.remove_node(t2)
#         self.task_id_map.pop(t2.id)
#         partition = sum(kernel.partition for kernel in t1.get_kernels())*1.0/len(t1.get_kernels())
#         if partition >= 5:
#             t1.set_partition(10)
#         else:
#             t1.set_partition(0)
#
#
#     def split_kernel_from_task(self, kernel, task):
#         """
#         Remove the given kernel from the given task and create a new task from that kernel, update task
#         dependencies accordingly. Returns the newly created task.
#         :param kernel:
#         :type kernel: Kernel
#         :param task:
#         :type task: Task
#         :return:
#         """
#         task.remove_kernel(kernel)
#         t = Task(kernel)
#         self.G.add_node(t)
#         self.tasks[kernel.id] = t
#         self.update_dependencies(task)
#         self.update_dependencies(t)
#         return t
#
#
#     def dispatch_single(self,priority  ,cmd_qs, ctxs, gpus, cpus ):
#
#             tsk = priority.description
#             #if(tsk.optm_device < 4 and False):
#             #    dev_type = 'gpu'
#             #else:
#             dev_type = 'cpu'
#             devices = set()
#             devices.add('gpu')
#             devices.add('cpu')
#             tsk.build_kernels( gpus, cpus, ctxs)
#             tsk.free_kernels = list()
#             a = []
#             b = []
#             for i in list(tsk.kernels):
#                 for j in self.free_kernels:
#                     if(i == self.get_kernel(j) and j not in self.finished_kernels and self.get_kernel(j) not in tsk.processing_kernel):
#                         tsk.free_kernels.append(self.get_kernel(j))
#
#             if(tsk.free_kernels):
#                 if(tsk not in self.device_to_task[dev_type].keys()):
#                     devices.remove(dev_type)
#                     if(tsk in self.device_to_task[list(devices)[0]].keys()):
#                         del self.device_to_task[dev_type][tsk]
#                     if(ready_queue[dev_type]):
#                         for i in list(ready_queue[dev_type]):
#                             if(i not in self.device_to_task[dev_type].values()):
#                                 self.device_to_task[dev_type][tsk] = i
#                                 break
#
#                 if(tsk in self.device_to_task[dev_type].keys()):
#                     if(self.device_to_task[dev_type][tsk] in ready_queue[dev_type]):
#                         ready_queue[dev_type].remove(self.device_to_task[dev_type][tsk])
#                         if(dev_type == 'gpu'):
#                             a, b = tsk.dispatch_single(self,self.device_to_task[dev_type][tsk] , -1, ctxs, cmd_qs, tsk)
#                         elif(dev_type == 'cpu'):
#                             a , b = tsk.dispatch_single(self,-1 ,self.device_to_task[dev_type][tsk] , ctxs, cmd_qs, tsk)
#
#             if(len(tsk.kernels) == len(tsk.finished_kernels)):
#                 if(tsk in self.device_to_task[dev_type].keys()):
#                     del self.device_to_task[dev_type][tsk]
#
#
#             if(len(tsk.finished_kernels) != len(tsk.kernels)):
#                 q.put(Skill(0, tsk , self , tsk.rank))
#
#             else:
#                 if(tsk in self.device_to_task[dev_type].keys()):
#                     del self.device_to_task[dev_type][tsk]
#             return a , b


class CLTrainer:

    def __init__(self, ex_map, global_map,filter_list=[],name=None):
        self.key_featvector_map = {}
        self.test_key_featvector_map = {}
        self.test_key_target_map = {}
        self.accuracy = 0.0
        self.sample_features = []
        self.sample_targets = []
        self.test_keys = filter_list
        self.key_list = []
        self.model = None
        self.name=name
        ex_cpu, ex_gpu = ex_map

        for key in global_map.keys():

            kernelName, worksize = key
            feat_dict = extract_feat_dict(key, global_map)
            extime = 0.0
            if ex_cpu[key] < ex_gpu[key]:
                extime = ex_cpu[key]
                feat_dict['Class']= "CPU"
            else:
                extime = ex_gpu[key]
                feat_dict['Class']= "GPU"


            feature_vector = self.get_feature_vector(feat_dict)

            if key not in filter_list:
                self.key_featvector_map[key] = feature_vector
                self.key_list.append(key)
                self.sample_features.append(np.asarray(feature_vector))
                self.sample_targets.append(feat_dict['Class'])

            else:
                self.test_key_featvector_map[key] = np.asarray(feature_vector)
                self.test_key_target_map[key] = feat_dict['Class']

    def feature_selection(self, num_features = 2):
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2

        self.sample_features = SelectKBest(chi2, k=num_features).fit_transform(self.sample_features, self.sample_targets)
        print "Reduced sample features shape", self.sample_features.shape

    def split_dataset(self, train_percentage):
        from sklearn.model_selection import train_test_split
        train_x, test_x, train_y, test_y = train_test_split(self.sample_features, self.sample_targets, train_size=train_percentage)
        return train_x, test_x, train_y, test_y

    def model_accuracy(self, train_x, test_x, train_y, test_y):
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import confusion_matrix

        predictions = self.model.predict(test_x)

        print "Training Accuracy: ", accuracy_score(train_y, self.model.predict(train_x))
        print "Testing Accuracy: ", accuracy_score(test_y, predictions)

    def model_cv_accuracy(self):
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(self.model, self.sample_features, self.sample_targets, cv=5)
        print "Cross Validation Accuracy: ", scores.mean(), "+/-", scores.std()*2

    def model_cv_accuracy_with_feature_selection(self, num_features):
        from sklearn.model_selection import cross_val_score
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import chi2
        from sklearn.feature_selection import f_classif

        for num_feat in range(2, num_features):
            reduced_features = SelectKBest(chi2, k=num_features).fit_transform(self.sample_features, self.sample_targets)
            scores = cross_val_score(self.model, reduced_features, self.sample_targets, cv=10)
            print "Cross Validation Accuracy for ",num_feat, "features", scores.mean(), "+/-", scores.std()*2

    def model_accuracy_with_smote(self):
        from imblearn.over_sampling import SMOTE
        from sklearn.model_selection import cross_val_score
        self.sample_features = np.asarray(self.sample_features)
        self.sample_targets = np.asarray(self.sample_targets)
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier()

        sm = SMOTE(random_state=12, ratio = 'minority')

        total_features = self.sample_features
        total_targets = self.sample_targets

        for key in self.test_key_featvector_map.keys():
            np.vstacK(total_features,self.test_key_featvector_map[key])
            np.vstack(total_targets, self.test_key_target_map[key])

        # x_res, y_res = sm.fit_sample(self.sample_features, self.sample_targets)

        x_res, y_res = sm.fit_sample(total_features, total_targets)

        itercount = 0
        delete_indices = []
        for element in x_res:
            for key in self.test_key_featvector_map.keys():
                if np.array_equal(element, self.test_key_featvector_map[key]):
                    delete_indices.append(itercount)
            itercount +=1
        x_res = np.delete(x_res,delete_indices,0)
        y_res = np.delete(y_res,delete_indices,0)

        scores = cross_val_score(self.model, x_res, y_res, cv=10)
        print "Cross Validation Accuracy: ", scores.mean(), "+/-", scores.std()*2


    def train_classifier(self,classifier_name):

        self.sample_features = np.asarray(self.sample_features)
        self.sample_targets = np.asarray(self.sample_targets)
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=12, ratio = 'minority')

        if classifier_name == "RandomForest":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=1000)
            total_features = self.sample_features
            total_targets = self.sample_targets

            for key in self.test_key_featvector_map.keys():
                total_features = np.vstack((total_features, self.test_key_featvector_map[key]))
                total_targets = np.append(total_targets, self.test_key_target_map[key])

            x_res, y_res = sm.fit_sample(total_features, total_targets)

            itercount = 0
            delete_indices = []
            for element in x_res:
                for key in self.test_key_featvector_map.keys():
                    if np.array_equal(element, self.test_key_featvector_map[key]):
                        delete_indices.append(itercount)
                itercount +=1

            x_res = np.delete(x_res,delete_indices,0)
            y_res = np.delete(y_res,delete_indices,0)
            # print x_res.shape
            # x_res, y_res = sm.fit_sample(self.sample_features, self.sample_targets)
            self.model.fit(x_res, y_res)
##############################################################


ALL_COLORS = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 'rgb(148, 103, 189)',
              'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 189, 34)', 'rgb(23, 190, 207)',
              'rgb(217,217,217)', 'rgb(240,2,127)', 'rgb(253,205,172)', 'rgb(179,205,227)', 'rgb(166,86,40)',
              'rgb(51,160,44)', 'rgb(247,129,191)', 'rgb(253,191,111)', 'rgb(190,186,218)', 'rgb(231,41,138)',
              'rgb(166,216,84)', 'rgb(153,153,153)', 'rgb(166,118,29)', 'rgb(230,245,201)', 'rgb(255,255,204)',
              'rgb(102,102,102)', 'rgb(77,175,74)', 'rgb(228,26,28)', 'rgb(217,95,2)', 'rgb(255,255,179)',
              'rgb(178,223,138)', 'rgb(190,174,212)', 'rgb(253,180,98)', 'rgb(255,217,47)', 'rgb(31,120,180)',
              'rgb(56,108,176)', 'rgb(229,216,189)', 'rgb(251,154,153)', 'rgb(222,203,228)', 'rgb(203,213,232)',
              'rgb(188,128,189)', 'rgb(55,126,184)', 'rgb(231,138,195)', 'rgb(244,202,228)', 'rgb(191,91,23)',
              'rgb(128,177,211)', 'rgb(27,158,119)', 'rgb(229,196,148)', 'rgb(253,218,236)', 'rgb(102,166,30)',
              'rgb(241,226,204)', 'rgb(255,127,0)', 'rgb(252,141,98)', 'rgb(227,26,28)', 'rgb(254,217,166)',
              'rgb(141,160,203)', 'rgb(204,235,197)', 'rgb(117,112,179)', 'rgb(152,78,163)', 'rgb(202,178,214)',
              'rgb(141,211,199)', 'rgb(106,61,154)', 'rgb(253,192,134)', 'rgb(255,255,51)', 'rgb(179,226,205)',
              'rgb(127,201,127)', 'rgb(251,128,114)', 'rgb(255,242,174)', 'rgb(230,171,2)', 'rgb(102,194,165)',
              'rgb(255,255,153)', 'rgb(179,179,179)', 'rgb(179,222,105)', 'rgb(252,205,229)', 'rgb(204,204,204)',
              'rgb(242,242,242)', 'rgb(166,206,227)', 'rgb(251,180,174)']

AC = ALL_COLORS


def plot_gantt_chart_graph(device_history, filename):
    """
    Plots Gantt Chart and Saves as png.

    :param device_history: Dictionary Structure containing timestamps of every kernel on every device
    :type device_history: dict
    :param filename: Name of file where the gantt chart is saved. The plot is saved in gantt_charts folder.
    :type filename: String
    """
    import random
    import colorsys
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def save_png(fig, filename):
        fig.savefig(filename)
        print "GANTT chart is saved at %s" % filename

    def get_N_HexCol(N=5):

        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in xrange(N)]
        hex_out = []
        for rgb in HSV_tuples:
            rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
            hex_out.append("".join(map(lambda x: chr(x).encode('hex'), rgb)))
        return hex_out

    def get_N_random_HexColor(N=5):
        HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in xrange(N)]
        hex_out = []
        'rgb(31, 119, 180)'
        indexs = random.sample(range(0, 77), N)
        for i in indexs:
            r = int(ALL_COLORS[i][4:-1].split(",")[0])
            g = int(ALL_COLORS[i][4:-1].split(",")[1])
            b = int(ALL_COLORS[i][4:-1].split(",")[2])
            hex_out.append('#%02x%02x%02x' % (r, g, b))
        return hex_out

    def list_from_file(file):
        device_info_list = []
        dev_data = open(file, "r")
        for line in dev_data:
            if "HOST_EVENT" in line:
                d_list = line.split(" ")[1:]
                device_info_list.append(d_list)
        return device_info_list

    def list_from_dev_history(dev_history):
        device_info_list = []
        for his in dev_history:
            device_info_list.append(his.split(" ")[1:])
        return device_info_list

    def get_min(device_info_list):
        g_min = Decimal('Infinity')
        for item in device_info_list:
            n = Decimal(min(item[3:], key=lambda x: Decimal(x)))
            if g_min > n:
                g_min = n
        return g_min

    def get_max(device_info_list):
        g_max = -1
        for item in device_info_list:
            x = Decimal(max(item[3:], key=lambda x: Decimal(x)))
            if g_max < x:
                g_max = x
        return g_max

    def normalise_timestamp(device_info_list):
        min_t = get_min(device_info_list)
        for item in device_info_list:
            for i in range(len(item) - 3):
                item[i + 3] = Decimal(item[i + 3]) - min_t
        return device_info_list

    device_info_list = normalise_timestamp(list_from_dev_history(device_history))

    colourMap = {}
    # colors = get_N_HexCol(len(device_info_list))
    colors = get_N_random_HexColor(len(device_info_list))

    c = 0
    dev_time = {}
    for k in device_info_list:
        kn = k[0] + "_" + k[1]

        kernel_times = [k[2], k[3], k[-1]]
        if kn not in dev_time:
            dev_time[kn] = []
        if kn in dev_time:
            dev_time[kn].append(kernel_times)

    for k in device_info_list:
        colourMap[k[2]] = colors[c]
        c = c + 1

    # legend_patches = []
    # for kn in colourMap:
    #     patch_color = "#" + colourMap[kn]
    #     legend_patches.append(patches.Patch(color=patch_color, label=str(k[2])))

    fig, ax = plt.subplots(figsize=(20, 10))
    device = 0
    #print dev_time
    for dev in dev_time:
        for k in dev_time[dev]:
            kname = k[0]
            # patch_color = "#" + colourMap[kname]
            patch_color = colourMap[kname]
            start = k[1]
            finish = k[2]
            y = 5 + device * 5
            x = start
            height = 5
            width = finish - start
            # print kname.split(",")[-1] + " : " + str(x) + "," + str(y) + "," + str(width) + "," + str(height)
            ax.add_patch(patches.Rectangle((x, y), width, height, facecolor=patch_color, edgecolor="#000000",
                                           label=kname.split(",")[-1]))
        device = device + 1
    plt.legend(loc=1)
    ax.autoscale(True)
    x_length = float(get_max(device_info_list))
    ax.set_xlim(0, 1.2 * x_length)
    ax.set_ylim(0, len(dev_time) * 10, True, True)
    labels = [item.get_text() for item in ax.get_yticklabels()]
    labels[0] = ""
    i = 1
    for dev in dev_time:
        labels[i] = (dev)
        i = i + 1

    y_ticks = np.arange(2.5, 2.5 + 5 * (1 + len(dev_time)), 5)

    plt.yticks(y_ticks.tolist(), labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('time ( in second )')
    ax.set_ylabel('devices')
    ax.set_yticklabels(labels)

    save_png(fig, filename)
