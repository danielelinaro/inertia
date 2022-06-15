#!/bin/bash

for config in area_1_config/*training*comp.json ; do
    logfile=`basename ${config%.json}.log`
    python3 build_data.py -s training_set -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1_comp $config > $logfile &
done
#wait

for config in area_1_config/*test*comp.json ; do
    logfile=`basename ${config%.json}.log`
    python3 build_data.py -s test_set -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1_comp $config > $logfile &
done
#wait

for config in area_1_config/*validation*comp.json ; do
    logfile=`basename ${config%.json}.log`
    python3 build_data.py -s validation_set -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1_comp $config > $logfile &
done
wait

#for config in area_1_config/compensators/*comp*.json ; do
#    logfile=`basename ${config%.json}.log`
#    python3 build_data.py -s compensator -o data/IEEE39/converted_from_PowerFactory/all_stoch_loads/var_H_area_1 $config > $logfile &
#done
#wait

