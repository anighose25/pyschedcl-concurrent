 python scheduling/create_transformer.py 6 0 16
 python scheduling/dynamic_heft_scheduler.py -f ./dag_info/dag_transformer/ -ng 1 -nc 1 -rc -thd -ef logs/transformer_profiling_64_64_64_with_delays.json -fdp logs/dheft_64_16_with_delays.json
 python scheduling/create_transformer.py 7 0 16
 python scheduling/dynamic_heft_scheduler.py -f ./dag_info/dag_transformer/ -ng 1 -nc 1 -rc -thd -ef logs/transformer_profiling_128_128_128_with_delays.json -fdp logs/dheft_128_16_with_delays.json
 python scheduling/create_transformer.py 8 0 16
 python scheduling/dynamic_heft_scheduler.py -f ./dag_info/dag_transformer/ -ng 1 -nc 1 -rc -thd -ef logs/transformer_profiling_256_256_256_with_delays.json -fdp logs/dheft_256_16_with_delays.json
 python scheduling/create_transformer.py 9 0 16
 python scheduling/dynamic_heft_scheduler.py -f ./dag_info/dag_transformer/ -ng 1 -nc 1 -rc -thd -ef logs/transformer_profiling_512_512_512_with_delays.json -fdp logs/dheft_512_16_with_delays.json
