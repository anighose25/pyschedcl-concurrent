#! /bin/bash

CLANG=clang
OPT=opt
LLVM_DIS=llvm-dis

INPUT_FILE=$1
KERNEL_NAME=$2

OCLDEF=/home/anirban/ResearchTools/Github/pyschedcl-stable/pyschedcl/passes/feature_pass/thrud/include/opencl_spir.h
$CLANG -v -x cl \
       -O0 \
       -target spir \
       -include ${OCLDEF} \
       ${INPUT_FILE} \
       -S -emit-llvm -fno-builtin -o - |
$OPT -instnamer \
     -mem2reg \
     -o - |
$LLVM_DIS -o -  
