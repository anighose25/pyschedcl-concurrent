import os
import sys
import subprocess
import json

def make_ffc_kernel(kernel_number,m1,n1,p1,partition,TS=32,wpt=1):
    return "{} coalesced_gemm.json {}\"m1\":{},\"p1\":{},\"n1\":{},\"TS\":{},\"wpt\":{},\"n_chunks\":1,\"partition\":{}{}\n".\
    format(kernel_number,"{",m1,p1,n1,TS,wpt,partition,"}")

def make_coalesced_transpose_kernel(kernel_number,m1,n1,partition):
    return "{} coalesced_transpose.json {}\"m1\":{},\"n1\":{},\"n_chunks\":1,\"partition\":{}{}\n".\
    format(kernel_number,"{",m1,n1,partition,"}")

def make_naive_softmax(kernel_number,m1,n1,partition):
    return "{} softmax.json {}\"r1\":{},\"c1\":{},\"n_chunks\":1,\"partition\":{}{}\n".\
    format(kernel_number,"{",m1,n1,partition,"}")

def make_empty_kernel(kernel_number,n1,w1,q1,partition):
    return "{} empty.json {}\"n1\":{},\"w1\":{},\"q1\":{},\"n_chunks\":1,\"partition\":{}{}\n".\
    format(kernel_number,"{",n1,w1,q1,partition,"}")



kernel="coalesced_gemm"
to_write = make_ffc_kernel("0",32,32,32,10)


dump_folder = "./profiling/dumps_transformer/"+kernel
dump_file_name = "timing.json"
if not os.path.exists(dump_folder):
    os.makedirs(dump_folder)

with open("./dag_info/dag_3_gemm/dag.graph","w") as f:
    f.write(to_write)
    f.write("---\n")
    f.write("---\n")
print(to_write)

subprocess.call("python scheduling/multiple_dag_devices.py -f ./dag_info/dag_3_gemm/ -ng 1 -nc 1 -rc -fdp {}/{}".format(dump_folder,dump_file_name),shell=True)
