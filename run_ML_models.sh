 #!/bin/bash

configs=("MLP" "random_forest" "kernel_ridge" "SVR" "nearest_neighbors")
nreps=(10 10 1 1 1)
nfiles=${#configs[@]}
let nfiles=nfiles-1
for i in `seq 0 $nfiles` ; do
    config=${configs[$i]}
    n=${nreps[$i]}
    for j in `seq $n` ; do
	logfile="${config}_${j}.log"
	python3 train_ML_model.py --no-comet config/ML/${config}.json > $logfile &
	sleep 2
    done
    if [ $n -gt 1 ] ; then
	wait
    fi
done
wait
