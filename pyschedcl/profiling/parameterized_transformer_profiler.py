import os
import sys
import subprocess
import json



def make_ffc_kernel(kernel_number,m1,n1,p1,partition,TS=32,wpt=1):
    return "{} FFC_sans_bias.json {}\"m1\":{},\"p1\":{},\"n1\":{},\"TS\":{},\"wpt\":{},\"n_chunks\":1,\"partition\":{}{}\n".\
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


kernel=sys.argv[1]#"coalesced_gemm"
partition = int(sys.argv[2])
N = int(sys.argv[3])
Q = int(sys.argv[4])
W = int(sys.argv[5])
run_number = 0
to_write = None





if kernel=="FFC_sans_bias":
    # to_write = make_ffc_kernel("0",global_work_size,global_work_size,global_work_size,partition)
    to_write = make_ffc_kernel("0",m1=N,n1=Q,p1=W,partition=partition)
if kernel=="softmax":
    # to_write = make_naive_softmax("0",global_work_size,global_work_size,partition)
    to_write = make_naive_softmax("0",m1=N,n1=N,partition=partition)
if kernel=="coalesced_transpose":
    # to_write = make_coalesced_transpose_kernel("0",global_work_size,global_work_size,partition)
    to_write = make_coalesced_transpose_kernel("0",m1=N,n1=Q,partition=partition)
if kernel=="empty":
    # to_write = make_empty_kernel("0",global_work_size,global_work_size,global_work_size,partition)
    to_write = make_empty_kernel("0",n1=N,w1=W,q1=Q,partition=partition)

dump_folder = "./profiling/dumps_transformer/"+kernel
dump_file_name = kernel+"_"+str(partition)+"_"+str(N)+"_"+str(Q)+"_"+str(W)+"_"+str(run_number)+".json"
if not os.path.exists(dump_folder):
    os.makedirs(dump_folder)

print to_write
# sys.exit(-1)

with open("./dag_info/dag_3_gemm/dag.graph","w") as f:
    f.write(to_write)
    f.write("---\n")
    f.write("---\n")
print to_write

subprocess.call("python scheduling/multiple_dag_devices.py -f ./dag_info/dag_3_gemm/ -ng 1 -nc 1 -rc -fdp {}/{}".format(dump_folder,dump_file_name),shell=True)
