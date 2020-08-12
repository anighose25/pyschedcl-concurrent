import numpy as np


def test(kernel_name,ref_ip,ref_op,ref_iop,dataset,verbose=False):
    print kernel_name
    if kernel_name == "mm" or kernel_name == "gemm":
        ###tested and working
        print "testing matrix multiplication"
        i1 = ref_ip[0].reshape([dataset,-1])
        i2 = ref_ip[1].reshape([dataset,-1])
        o_pred = ref_op[0].reshape([dataset,-1])
        o_act = i1.dot(i2)
        #print o_pred
        #print o_act
        diff = np.mean(np.abs(o_pred-o_act))
        return diff < 1e-4

    elif kernel_name == "atax_kernel1":
        print "testing matrix vector multiplication"
        i1 = ref_ip[0].reshape([dataset,-1])
        i2 = ref_ip[1].reshape([dataset])
        print "A\n",i1
        print "B\n",i2
        o_pred = ref_op[0].reshape([dataset])
        o_act = i1.dot(i2)
        print "output - pyschedcl", o_pred
        print "output - numpy", o_act
        diff = np.mean(np.abs(o_pred-o_act))
        print diff
        return diff < 1e-4

    else:
        if verbose:
            for i,ip in enumerate(ref_ip):
                print "input ",i," ",ip

            print "\n"

            for i,op in enumerate(ref_op):
                print "output ",i, " ",op

            print "\n"

            for i,op in enumerate(ref_iop):
                print "io ",i, " ",op
