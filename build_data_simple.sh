#!/bin/bash

for config in area_1_config_grid/training*.json ; do
    tmp=`basename $config`
    num="${tmp:13:2}"
    logfile=`basename ${config%.json}.log`
    if [ "$num" = "01" ] || [ "$num" = "02" ] || [ "$num" = "10" ] || [ "$num" = "11" ] ; then
	python3 build_data.py -s training_set -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1_comp_grid $config > $logfile &
    fi
done
wait

for config in area_1_config_grid/test*.json ; do
    tmp=`basename $config`
    num="${tmp:9:2}"
    logfile=`basename ${config%.json}.log`
    if [ "$num" = "01" ] || [ "$num" = "02" ] || [ "$num" = "10" ] || [ "$num" = "11" ] ; then
	python3 build_data.py -s test_set -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1_comp_grid $config > $logfile &
    fi
done
wait

for config in area_1_config_grid/validation*.json ; do
    tmp=`basename $config`
    num="${tmp:15:2}"
    logfile=`basename ${config%.json}.log`
    if [ "$num" = "01" ] || [ "$num" = "02" ] || [ "$num" = "10" ] || [ "$num" = "11" ] ; then
	python3 build_data.py -s validation_set -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1_comp_grid $config > $logfile &
    fi
done
wait

#for config in area_1_config/compensators/*comp*.json ; do
#    logfile=`basename ${config%.json}.log`
#    python3 build_data.py -s compensator -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1 $config > $logfile &
#done
#wait

