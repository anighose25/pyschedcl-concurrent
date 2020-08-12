#! /bin/bash

CLANG=clang
OPT=opt
LLVM_DIS=llvm-dis
AXTOR=axtor
LIB_THRUD=/home/anirban/Tools/Compilers/llvm-analysis-tools/feature_extraction/build/thrud/lib/libThrud.so
OCL_DEF=/home/anirban/Tools/Compilers/llvm-analysis-tools/feature_extraction/coarsening_pass/thrud/include/opencl_spir.h
TARGET=spir

INPUT_FILE=$1
KERNEL_NAME=$2
OUTPUT_FILE=$3
OPTIMIZATION=-O0

$CLANG -x cl \
       -target $TARGET \
       -include $OCL_DEF \
       ${OPTIMIZATION} \
       ${INPUT_FILE} \
       -S -emit-llvm -fno-builtin -o - |
$OPT   -instnamer -mem2reg -load $LIB_THRUD -structurizecfg -opencl-loop-instcount -count-loop-kernel-name ${KERNEL_NAME}>/dev/null 2>${OUTPUT_FILE} 
#${LLVM_DIS} -o -  

#${OPT} -dot-cfg-only
