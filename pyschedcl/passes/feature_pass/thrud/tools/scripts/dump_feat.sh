#! /bin/bash

CLANG=clang
OPT=opt
LLVM_DIS=llvm-dis

INPUT_FILE=$1
KERNEL_NAME=$2

OCLDEF=/home/anirban/Tools/Compilers/llvm-analysis-tools/feature_extraction/coarsening_pass/thrud/include/opencl_spir.h

$CLANG -x cl \
       -O0 \
       -target spir \
       -include ${OCLDEF} \
       ${INPUT_FILE} \
       -S -emit-llvm -fno-builtin -o - |
$OPT -instnamer \
     -mem2reg \
     -load $LIBTHRUD -opencl-instcount \
     -o - |
$LLVM_DIS -o -  
