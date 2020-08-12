#!/bin/bash
# Basic while loop
counter=1
while [ $counter -le 100 ]
do
	python scheduling/multiple_dag_devices.py -f dag_info/dag_3_gemm/ -ng 2 -nc 1 -rc
	((counter++))
done
echo All done
