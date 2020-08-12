#! /bin/bash

CLANG=clang
OPT=opt
LLVM_DIS=llvm-dis
AXTOR=axtor
LIB_THRUD=/home/anirban/Tools/Compilers/llvm-analysis-tools/feature_extraction/build/thrud/lib/libThrud.so
OCL_DEF=/home/anirban/Tools/Compilers/llvm-analysis-tools/feature_extraction/coarsening_pass/thrud/include/opencl_spir.h
TARGET=spir

INPUT_FILE=$1
OPTIMIZATION=-O0

$CLANG -x cl \
       -target $TARGET \
       -include $OCL_DEF \
       ${OPTIMIZATION} \
       ${INPUT_FILE} \
       -S -emit-llvm -fno-builtin -o test.bc
$OPT -load $LIB_THRUD -opencl-instcount -analyze\
     < test.bc 
#$OPT -load $LIB_THRUD -opencl-instcount <reduce_test.bc > /dev/null
#${LLVM_DIS} -o -  

#${OPT} -dot-cfg-only
