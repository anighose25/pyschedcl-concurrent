#!/bin/bash
# rm -rf ./profiling/dumps_transformer/paramterised_dag_profiling/
# mkdir ./profiling/dumps_transformer/paramterised_dag_profiling/
# for (( size = 6; size <= 11;size++ ))      ### Outer for loop ###
# do
#     for (( cqgpu = 1 ; cqgpu <= 5; cqgpu++ )) ### Inner for loop ###
#     do
#
#         for (( headoncpu = 0 ; headoncpu <= 8; headoncpu++ ))
#         do
#           python scheduling/create_transformer.py $size $headoncpu 8
#           python scheduling/multiple_dag_devices.py -ng $cqgpu -nc $cqgpu -rc -fdp './profiling/dumps_transformer/paramterised_dag_profiling/'$size'_GPU'$cqgpu'_CPU'$cqgpu'_'$headoncpu'.json'
#           #echo './profiling/dumps_transformer/paramterised_dag_profiling/'$size'_GPU'$cqgpu'_CPU'$cqcpu'_'$headoncpu'.json'
#           #echo "\n"
#         done
#
#     done
# done

# for (( cqgpu = 1 ; cqgpu <= 5; cqgpu++ ))
# do
#
#     for (( headoncpu = 0 ; headoncpu <= 8; headoncpu++ ))
#     do
#       python scheduling/create_transformer.py 'default' $headoncpu 8
#       python scheduling/multiple_dag_devices.py -ng $cqgpu -nc $cqgpu -rc -fdp './profiling/dumps_transformer/paramterised_dag_profiling/default_GPU'$cqgpu'_CPU'$cqgpu'_'$headoncpu'.json'
#       #echo './profiling/dumps_transformer/paramterised_dag_profiling/'$size'_GPU'$cqgpu'_CPU'$cqcpu'_'$headoncpu'.json'
#       #echo "\n"
#     done
#
# done
# rm -rf ./profiling/dumps_transformer/parameterised_dag_16_heads_multiple_sizes/
mkdir ./profiling/dumps_transformer/parameterised_dag_16_heads_multiple_sizes/
#for (( trial = 0 ; trial < 5 ;trial++ ))      ### Num trial loop ###
#do
for (( size = 6 ; size <= 9 ;size++ ))      ### Outer for loop ###
do
    for (( cqgpu = 1 ; cqgpu <= 5; cqgpu++ )) ### Inner for loop ###
    do
       for(( cqcpu = 1 ; cqcpu <= 5; cqcpu++ ))
       do
          for (( headoncpu = 0 ; headoncpu <= 16; headoncpu++ ))
          do
            python scheduling/create_transformer.py $size $headoncpu 16 
            python scheduling/multiple_dag_devices.py -f ./dag_info/dag_transformer/ -ng $cqgpu -nc $cqcpu -rc -fdp './profiling/dumps_transformer/parameterised_dag_1_16_heads_multiple_runs/'$size'_GPU'$cqgpu'_CPU'$cqcpu'_'$headoncpu'.json'
            echo './profiling/dumps_transformer/parameterised_dag_profiling/'$size'_GPU'$cqgpu'_CPU'$cqcpu'_'$headoncpu'.json'
            echo "\n"
          done
        done
    done
done
#done