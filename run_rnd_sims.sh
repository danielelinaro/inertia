#!/bin/bash

N=20
outdir="data/IEEE39/converted_from_PowerFactory/all_stoch_loads/high_momentum_test_comp"
#G02m=2.825
#G03m=3.153
#G02m=2.932
#G03m=3.247
#G02m=6.039
#G03m=5.966
#G02m=6.146
#G03m=6.059
#G02s=0.1
#G03s=0.1
Comp1m=9
Comp1s=0.1

#G02H=( $(python3 normal.py $G02m $G02s $N) )
#G03H=( $(python3 normal.py $G03m $G03s $N) )
Comp1H=( $(python3 normal.py $Comp1m $Comp1s $N) )
inconfig="IEEE39_PF_sim_config_rnd.json"
outconfig="rnd_config_high_comp.json"

mkdir -p $outdir

let N=$N-1
for i in `seq 0 $N` ; do
    #sed -e 's/#G02H#/'${G02H[$i]}'/' -e 's/#G03H#/'${G03H[$i]}'/' $inconfig > $outconfig
    sed -e 's/#Comp1H#/'${Comp1H[$i]}'/' $inconfig > $outconfig
    python3 run_simulation.py -O $outdir $outconfig
done
