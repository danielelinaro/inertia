#!/bin/bash

nreps=1
for cfgfile in config/ML/*.json ; do
    for i in `seq $nreps` ; do
	logfile="${cfgfile%.json}_${i}.log"
	python3 train_ML_model.py --no-comet $cfgfile > $logfile &
	sleep 2
    done
    #wait
done
wait
