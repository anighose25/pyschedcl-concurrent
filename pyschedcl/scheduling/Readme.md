# create_transformer.py 
This script is used to create a Transformer DAG specification file for Pyschedcl. Note that the size of the individual kernels is the 2's exponent of the size parameter below. Individual functions of the script are documented within the file itself.
```
python create_transfomer.py <size> <num_heads_on_cpu> <total_num_heads>
```

# multiple_dag_devices.py 
This script is used to run the Pyschedcl runtime. For example to run pyschedcl on the output of create_transformer.py 
```
python multiple_dag_devices.py -f <addr of dag spec file> 
                                -nc <number of cpus> 
                                -ng <number of gpus> 
                                -thd <use threaded top level scheduler>
                                -rc <recreate dag from spec file>

```

# heft_scheduler.py
TODO




