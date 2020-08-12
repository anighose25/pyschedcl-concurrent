import os
import sys

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

def make_one_head(base=0,N=64,W=512,Q=64,partition=10):
    kernel_defs = [None]*9
    deps = []
    for i in range(3):
        kernel_defs[i] = make_ffc_kernel(kernel_number=base+i,m1=N,n1=Q,p1=W,partition=partition)
    kernel_defs[3] = make_coalesced_transpose_kernel(kernel_number=base+3,m1=N,n1=Q,partition=partition)
    kernel_defs[4] = make_ffc_kernel(kernel_number=base+4,m1=N,n1=N,p1=Q,partition=partition)
    kernel_defs[5] = make_naive_softmax(kernel_number=base+5,m1=N,n1=N,partition=partition)
    kernel_defs[6] = make_ffc_kernel(kernel_number=base+6,m1=N,n1=Q,p1=N,partition=partition)
    kernel_defs[7] = make_ffc_kernel(kernel_number=base+7,m1=N,n1=W,p1=Q,partition=partition)
    kernel_defs[8] = make_empty_kernel(kernel_number=base+8,n1=N,w1=W,q1=Q,partition=partition)


    deps.append("{} {}->{} {}\n".format(base+0,2,base+4,0))
    deps.append("{} {}->{} {}\n".format(base+1,2,base+3,0))
    deps.append("{} {}->{} {}\n".format(base+3,1,base+4,1))
    deps.append("{} {}->{} {}\n".format(base+4,2,base+5,0))
    deps.append("{} {}->{} {}\n".format(base+2,2,base+6,1))
    deps.append("{} {}->{} {}\n".format(base+5,1,base+6,0))
    deps.append("{} {}->{} {}\n".format(base+6,2,base+7,0))

    deps.append("{} {}->{} {}\n".format(base+8,0,base+0,0))


    deps.append("{} {}->{} {}\n".format(base+8,0,base+1,0))

    deps.append("{} {}->{} {}\n".format(base+8,0,base+2,0))


    return kernel_defs,deps


kernels = []
deps = []

total_heads = int(sys.argv[3])
total_heads_on_cpu = int(sys.argv[2])
size = 2**(int(sys.argv[1]))

for i in range(total_heads):
    part=10
    if i < total_heads_on_cpu:
        part=0

    k,d = make_one_head(N=size,W=size,Q=size,base=9*i,partition=part)
    kernels += k
    deps += d

with open("./dag_info/dag_transformer/dag.graph","w") as f:
    for kernel in kernels:
        f.write(kernel)
    f.write("---\n")
    for dep in deps:
        f.write(dep)
    f.write("---\n")
