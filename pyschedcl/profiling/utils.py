from copy import deepcopy

def adjust_zero(timestamps):
    kernels = timestamps.keys()
    reference_device = {}
    reference_host = {}
    total_time = 0
    global_reference = None

    for kernel in kernels:
        print "Adjusting for ",kernel
        device = timestamps[kernel]["device"]
        if device == "gpu":
            t = timestamps[kernel]["write"]["device_queued"]
        else:
            t = timestamps[kernel]["nd_range"]["device_start"]
        if t == -1:
            continue

        if not (device in reference_device):
            reference_device[device] = t
        else:
            reference_device[device] = min(reference_device[device],t)

        t = timestamps[kernel]["write"]["host_queued_end"]

        if not (device in reference_host):
            reference_host[device] = t
        else:
            reference_host[device] = min(reference_host[device],t)

        t = timestamps[kernel]["write"]["host_queued_start"]

        if not global_reference:
            global_reference = t
        else:
            global_reference = min(global_reference,t)


    relative_timestamps = deepcopy(timestamps)

    global_reference = None

    for key,value in reference_host.items():
        if not global_reference:
            global_reference = value
        else:
            global_reference = min(value,global_reference)


    for kernel,kernel_timestamps in relative_timestamps.items():
        device = kernel_timestamps["device"]
        for event_type,event_timestamps in kernel_timestamps.items():
            #print(event_type)
            if event_type in ["device","cmdq"]:
                continue
            else:
                #continue
                for sub_event_type in event_timestamps:
                    if  sub_event_type[:4] == "host":
                        event_timestamps[sub_event_type] -= global_reference
                        continue
                    else:
                        event_timestamps[sub_event_type] = event_timestamps[sub_event_type] - reference_device[device] + reference_host[device] - global_reference
                        total_time = max(total_time,event_timestamps[sub_event_type])

    #print "Total Time Taken - ",total_time
    #print(json.dumps(relative_timestamps,sort_keys=True,indent=1))
    return relative_timestamps, total_time
