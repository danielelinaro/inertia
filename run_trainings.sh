#!/bin/bash

script="train_network.py"
config_dir="area_1_config/DZA=0.0"
subdirs=("without_ReLU" "with_ReLU")
#config_files=("bus_3" "bus_14" "bus_17")
#num_cores=(12 12 12)
#config_files=("bus_3-14" "bus_3-17" "bus_14-17")
#num_cores=(12 12 12)
#config_files=("bus_3-14-17")
#num_cores=(32)

config_files=("bus_3" "bus_14" "bus_17" "bus_3-14" "bus_3-17" "bus_14-17" "bus_3-14-17")
num_cores=(8 8 8 8 8 8 8)

for subdir in ${subdirs[@]} ; do
    for i in ${!config_files[@]} ; do
	stop=3
	cores=${num_cores[$i]}
	if [ "$subdir" = "without_ReLU" -a \
		       \( "${config_files[$i]}" = "bus_3-14" -o \
		          "${config_files[$i]}" = "bus_3-14-17" \) ] ; then
	    stop=2
	    cores=12
	fi
	for run in `seq 1 $stop` ; do
	    config_file=${config_files[$i]}
	    python3 $script --max-cores ${cores} ${config_dir}/${subdir}/${config_file}.json > ${config_file}_${run}.log &
	done
	wait
    done
done

