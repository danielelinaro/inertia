#!/bin/bash

subset="yes"

for config in area_1_config_grid/training*.json ; do
    logfile=`basename ${config%.json}.log`
    if [ "$subset" = "yes" ] ; then
	tmp=`basename $config`
	num="${tmp:9:3}"
	if [ "$num" = "001" ] || [ "$num" = "002" ] || [ "$num" = "012" ] || [ "$num" = "013" ] \
	       || [ "$num" = "109" ] || [ "$num" = "110" ] || [ "$num" = "120" ] || [ "$num" = "121" ] ; then
	    python3 build_data.py -s training_set -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1_comp_grid $config > $logfile &
	fi
    else
	python3 build_data.py -s training_set -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1_comp_grid $config > $logfile &
    fi
done
wait

for config in area_1_config_grid/test*.json ; do
    logfile=`basename ${config%.json}.log`
    if [ "$subset" = "yes" ] ; then
	tmp=`basename $config`
	num="${tmp:5:3}"
	if [ "$num" = "001" ] || [ "$num" = "002" ] || [ "$num" = "012" ] || [ "$num" = "013" ] \
	       || [ "$num" = "109" ] || [ "$num" = "110" ] || [ "$num" = "120" ] || [ "$num" = "121" ] ; then
	    python3 build_data.py -s test_set -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1_comp_grid $config > $logfile &
	fi
    else
	python3 build_data.py -s test_set -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1_comp_grid $config > $logfile &
    fi
done
wait

for config in area_1_config_grid/validation*.json ; do
    logfile=`basename ${config%.json}.log`
    if [ "$subset" = "yes" ] ; then
	tmp=`basename $config`
	num="${tmp:11:3}"
	if [ "$num" = "001" ] || [ "$num" = "002" ] || [ "$num" = "012" ] || [ "$num" = "013" ] \
	       || [ "$num" = "109" ] || [ "$num" = "110" ] || [ "$num" = "120" ] || [ "$num" = "121" ] ; then
	    python3 build_data.py -s validation_set -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1_comp_grid $config > $logfile &
	fi
    else
	python3 build_data.py -s validation_set -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1_comp_grid $config > $logfile &
    fi
done
wait

#for config in area_1_config/compensators/*comp*.json ; do
#    logfile=`basename ${config%.json}.log`
#    python3 build_data.py -s compensator -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1 $config > $logfile &
#done
#wait

