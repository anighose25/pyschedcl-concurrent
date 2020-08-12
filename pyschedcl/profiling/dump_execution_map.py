import subprocess
import os
import json
from copy import deepcopy
from collections import defaultdict
from utils import adjust_zero
import pandas as pd
import numpy as np


def construct_table(profiling_results,kernel,data=None):
	if not data:
		data = defaultdict(str)
	dev_dict = {0:"cpu",10:"gpu"}
	h2d = []
	d2h = []

	with open("./database/info/{}.json".format(kernel)) as f:
		source_info = json.load(f)
	workdim = source_info["workDimension"]

	for key in ["inputBuffers",'ioBuffers']:
	    if key in source_info:
	        for buff in source_info[key]:
	            h2d.append(buff['size'])

	for key in ["outputBuffers",'ioBuffers']:
	    if key in source_info:
	        for buff in source_info[key]:
	            d2h.append(buff['size'])
	print profiling_results
	# for result in profiling_results:
	result = profiling_results
	print "Opening", result
	f = open(result)
	timestamps = json.load(f)
	f.close()

	launch_params = result[:-5].split('_')
	print "launch parameters",launch_params
	device = int(launch_params[-5])
	print "value of device", device
	N = int(launch_params[-4])
	W = int(launch_params[-3])
	Q = int(launch_params[-2])
	c1 = r1 = w1 = q1 = m1 = p1 = n1 = 0 #dataset = global_work_size = int(launch_params[-3])
	if launch_params[0] == "coalesced":
		if launch_params[1]=="gemm":
			m1=N
			n1=Q
			p1=W
		else:
			m1=N
			n1=Q
	if launch_params[0] == "FFC":
			m1=N
			n1=Q
			p1=W

	if launch_params[0] == "softmax":
		r1=N
		c1=N
	if launch_params[0] == "empty":
		n1=N
		w1=W
		q1=Q



   	run_number = int(launch_params[-1])
   	h2d_num_bytes = sum([eval(buff_size) for buff_size in h2d])
   	d2h_num_bytes = sum([eval(buff_size) for buff_size in d2h])


	print(timestamps)
   	timestamps,total_time = adjust_zero(timestamps)

   	key = timestamps.keys()[0]

   	h2d_time = timestamps[key]["write"]["device_end"] - timestamps[key]["write"]["device_start"]
   	d2h_time = timestamps[key]["read"]["device_end"] - timestamps[key]["read"]["device_start"]
   	exec_time = timestamps[key]["nd_range"]["device_end"] - timestamps[key]["nd_range"]["device_start"]
   	delay = timestamps[key]["write"]["host_queued_end"]-timestamps[key]["write"]["host_queued_start"]
   	if device == 10:
   	    data["h2d_bytes"] = h2d_num_bytes
   	    data["d2h_bytes"] = d2h_num_bytes
   	    data["h2d_time"] = h2d_time
   	    data["d2h_time"] = d2h_time
   	    data["exec_gpu"] = exec_time
    	data["total_time_gpu"] = total_time
    	data['gpu_delay']=delay
	if device == 0:
		data["exec_cpu"] = exec_time
    	data["total_time_cpu"] = total_time
    	data['cpu_delay']=delay
	# print data.keys(),data.values()
	return data


if __name__ == '__main__':

	command_header = "python profiling/parameterized_transformer_profiler.py"
	kernels = ["FFC_sans_bias","coalesced_transpose","softmax","empty"]
	N=[64,128,256,512]
	Q=[64,128,256,512]
	W=[64,128,256,512]
	partition = [0, 10]
	profiling_enabled = False
	if profiling_enabled:
		for kernel in kernels:
			for n,q,w in zip(N,Q,W):
				for p in partition:
					run_command = command_header+ " " + kernel+ " " + str(p) + " " + str(n) + " " +str(q) + " " +str(w)
					print run_command
					subprocess.call(run_command,shell=True)
	# kernel_extime_map = defaultdict(lambda: defaultdict (lambda: defaultdict()))
	

	kernel_extime_map = defaultdict(str)
	

	#kernels = os.listdir("./profiling/dumps_transformer/")
	for n,q,w in zip(N,Q,W):
		for kernel in kernels:
			# profiling_results = os.listdir(os.path.join("./profiling/dumps_transformer",kernel))
			# profiling_results = "./profiling/dumps_transformer/"+kernel+"/"+kernel+
			print "Dumping profile statistics for ",kernel
	 		profiling_results_cpu = "./profiling/dumps_transformer/"+kernel+"/"+kernel+"_"+"0_"+str(n)+ "_" +str(q) + "_" +str(w) +"_0.json"
	 		# kernel_extime_map[kernel][global_work_size[0]][local_work_size[0]] = construct_table(profiling_results,kernel,data=None)
	 		kernel_extime_map[kernel] = construct_table(profiling_results_cpu, kernel)
	 		data = kernel_extime_map[kernel]
	 		profiling_results_gpu = "./profiling/dumps_transformer/"+kernel+"/"+kernel+"_"+"10_"+str(n)+ "_" +str(q) + "_" +str(w) +"_0.json"
	 		kernel_extime_map[kernel] = construct_table(profiling_results_gpu, kernel,data)
 	# print kernel_extime_map
	 	dump_transformer_data_file = "./logs/transformer_profiling_" + str(n)+ "_" +str(q) + "_" +str(w) +"_with_delays.json"
		with open(dump_transformer_data_file, 'w') as outfile:
	 		json.dump(kernel_extime_map, outfile)
#This is the kernel json file
