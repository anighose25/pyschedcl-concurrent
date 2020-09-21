# pyschedcl-concurrent
A Scheduling Framework that leverages concurrency in data parallel systems. This is a separate branch of the original codebase (PySchedCL)[https://github.com/anighose25/pyschedcl] with support for DAG scheduling and concurrency aware cluster scheduling. 

Overview
============

PySchedCL is a python based scheduling framework for OpenCL applications. The framework heavily relies on the PyOpenCL package which provides easy access to the full power of the OpenCL API to the end user. We present only the dependencies of the package and a one line description for every important folder and file present in the project. Detailed usage and documentation regarding the project may be found here.

Dependencies
------------------

+ Python 2.7
+ PIP
+ OpenCL Runtime Device Drivers (Minimum 1.2)
  - Intel
  - AMD
  - NVIDIA
+ PyOpenCL
+ gcc >=4.8.2


Project Hierarchy
-----------------

<pre>
<code>

└── <b>pyschedcl</b> (Base Package Folder)
    ├── <b>pyschedcl.py</b> (Base Package API)
    ├── <b>partition</b> (Folder containing scripts for partitioning)
    ├── <b>scheduling</b> (Folder containing scripts for scheduling)
    ├── <b>utils</b> (Folder containing additional utility scripts)
    ├── <b>database</b> (Folder containing kernel source files and kernel level json files used by framework)
    ├── <b>dag_info</b> (Folder containing  DAG json files used by framework)
    ├── <b>logs</b> (Folder containing timing logs and dumps for various data parallel applications)
    ├── <b>profiling</b> (Folder containing python notebooks for evaluating performance of different applications)
    ├── <b>profiling</b> (Folder containing python notebooks for evaluating performance of different applications)
    
  </code>
  </pre>
