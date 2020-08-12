declare -a arr=(
                "covar_kernel"  "syr2k_kernel"  "mt"
              "Convolution3D_kernel"  "Convolution2D_kernel"  "naive_kernel"  )


for i in "${arr[@]}"
do
   python profiling/profiler.py -f database/info/$i.json
done
