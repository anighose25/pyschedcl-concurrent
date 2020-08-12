#! /bin/bash

CLANG=clang
OPT=opt
LLVM_DIS=llvm-dis
AXTOR=axtor
LIB_THRUD=/home/anirban/ResearchTools/Github/pyschedcl-stable/pyschedcl/passes/build/thrud/lib/libThrud.so
OCL_DEF=/home/anirban/ResearchTools/Github/pyschedcl-stable/pyschedcl/passes/feature_pass/thrud/include/opencl_spir.h

TARGET=spir

INPUT_FILE=$1
KERNEL_NAME=$2
OUTPUT_FILE=$3
OPTIMIZATION=-O0

$CLANG -v -x cl \
       -target $TARGET \
       -include $OCL_DEF \
       ${OPTIMIZATION} \
       ${INPUT_FILE} \
       -S -emit-llvm -fno-builtin -o - |
$OPT   -instnamer -mem2reg -loop-simplify -load $LIB_THRUD -structurizecfg -opencl-instcount -count-kernel-name ${KERNEL_NAME}>/dev/null 2> ${OUTPUT_FILE} 
#${LLVM_DIS} -o -  

#${OPT} -dot-cfg-only
