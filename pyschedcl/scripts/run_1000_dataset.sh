for i in {1..1}
do
    echo -e "0 sample_$i.json {\"dataset\":64,\"partition\":10}\n---\n---" > ./dag_info/dag_transformer/dag.graph
    python scheduling/multiple_dag_devices.py -ng 1 -nc 1 -rc
done