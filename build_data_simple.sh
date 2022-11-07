#!/bin/bash

subset="no"
outdir="data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1_comp_grid/coarse"
confdir="area_1_config_grid_coarse/H_comp11_5.0/diagonal"

i=0
for config in ${confdir}/training*.json ; do
    logfile=`basename ${config%.json}.log`
    if [ "$subset" = "yes" ] ; then
	tmp=`basename $config`
	num="${tmp:9:3}"
	if [ "$num" = "001" ] || [ "$num" = "002" ] || [ "$num" = "012" ] || [ "$num" = "013" ] \
	       || [ "$num" = "109" ] || [ "$num" = "110" ] || [ "$num" = "120" ] || [ "$num" = "121" ] ; then
	    python3 build_data.py -s training_set -o $outdir $config > $logfile &
	    sleep 10
	fi
    else
	python3 build_data.py -s training_set -o $outdir $config > $logfile &
	sleep 10
    fi
    let i=i+1
    if [ $i -eq 12 ] ; then
	i=0
	wait
    fi
done
wait

i=0
for config in ${confdir}/test*.json ; do
    logfile=`basename ${config%.json}.log`
    if [ "$subset" = "yes" ] ; then
	tmp=`basename $config`
	num="${tmp:5:3}"
	if [ "$num" = "001" ] || [ "$num" = "002" ] || [ "$num" = "012" ] || [ "$num" = "013" ] \
	       || [ "$num" = "109" ] || [ "$num" = "110" ] || [ "$num" = "120" ] || [ "$num" = "121" ] ; then
	    python3 build_data.py -s test_set -o $outdir $config > $logfile &
	    sleep 10
	fi
    else
	python3 build_data.py -s test_set -o $outdir $config > $logfile &
	sleep 10
    fi
    let i=i+1
    if [ $i -eq 12 ] ; then
	i=0
	wait
    fi
done
wait

i=0
for config in ${confdir}/validation*.json ; do
    logfile=`basename ${config%.json}.log`
    if [ "$subset" = "yes" ] ; then
	tmp=`basename $config`
	num="${tmp:11:3}"
	if [ "$num" = "001" ] || [ "$num" = "002" ] || [ "$num" = "012" ] || [ "$num" = "013" ] \
	       || [ "$num" = "109" ] || [ "$num" = "110" ] || [ "$num" = "120" ] || [ "$num" = "121" ] ; then
	    python3 build_data.py -s validation_set -o $outdir $config > $logfile &
	    sleep 10
	fi
    else
	python3 build_data.py -s validation_set -o $outdir $config > $logfile &
	sleep 10
    fi
    let i=i+1
    if [ $i -eq 12 ] ; then
	i=0
	wait
    fi
done
wait

