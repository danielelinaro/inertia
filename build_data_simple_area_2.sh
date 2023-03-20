#!/bin/bash

subset="yes"
outdir="data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_2_comp_grid/coarse"
confdir="area_2_config_grid_coarse/H_comp21_0.1"
ncores=36

i=0
for config in ${confdir}/training*.json ; do
    logfile=`basename ${config%.json}.log`
    if [ "$subset" = "yes" ] ; then
	tmp=`basename $config`
	num="${tmp:9:2}"
	if [ "$num" = "01" ] || [ "$num" = "16" ] ; then
	    python3 build_data.py -s training_set -o $outdir $config > $logfile &
	    sleep 10
	fi
    else
	python3 build_data.py -s training_set -o $outdir $config > $logfile &
	sleep 10
    fi
    let i=i+1
    if [ $i -eq $ncores ] ; then
	i=0
	wait
    fi
done

if [ "$subset" = "no" ] ; then
    echo "Waiting..."
    wait
    i=0
fi
    
for config in ${confdir}/test*.json ; do
    logfile=`basename ${config%.json}.log`
    if [ "$subset" = "yes" ] ; then
	tmp=`basename $config`
	num="${tmp:5:2}"
	if [ "$num" = "01" ] || [ "$num" = "16" ] ; then
	    python3 build_data.py -s test_set -o $outdir $config > $logfile &
	    sleep 10
	fi
    else
	python3 build_data.py -s test_set -o $outdir $config > $logfile &
	sleep 10
    fi
    let i=i+1
    if [ $i -eq $ncores ] ; then
	i=0
	wait
    fi
done

if [ "$subset" = "no" ] ; then
    echo "Waiting..."
    wait
    i=0
fi

for config in ${confdir}/validation*.json ; do
    logfile=`basename ${config%.json}.log`
    if [ "$subset" = "yes" ] ; then
	tmp=`basename $config`
	num="${tmp:11:2}"
	if [ "$num" = "01" ] || [ "$num" = "16" ] ; then
	    python3 build_data.py -s validation_set -o $outdir $config > $logfile &
	    sleep 10
	fi
    else
	python3 build_data.py -s validation_set -o $outdir $config > $logfile &
	sleep 10
    fi
    let i=i+1
    if [ $i -eq $ncores ] ; then
	i=0
	wait
    fi
done
wait

