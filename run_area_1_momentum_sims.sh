#!/bin/bash

# this script changes the values of inertia of G2 and G3 to keep the area momentum fixed

N=10
reps=200
outdir="data/IEEE39/converted_from_PowerFactory/all_stoch_loads/low_momentum_test_var_G1_G2"
# default values of inertia of G2, G3 and the compensator in area 1
G02Hdef=2.932
G03Hdef=3.247
#G02Hdef=6.146
#G03Hdef=6.059
Comp11H=0.1
# values of nominal power of G2, G3 and the compensator in area 1
G02S=700
G03S=800
Comp11S=100
# defalut value of area momentum
Mdef=`echo "($G02Hdef*$G02S+$G03Hdef*$G03S+$Comp11H*$Comp11S)/30*0.001" | bc -l`

inconfig="IEEE39_PF_sim_config.json"
outconfig="tmp_low.json"

mkdir -p $outdir

G02Hlow=2
G02Hhigh=4
#G02Hlow=5
#G02Hhigh=7
G02H=( $(python3 linspace.py $G02Hlow $G02Hhigh $N) )
for g02h in ${G02H[@]} ; do
    g03h=`printf %.3f $(echo "($Mdef*1000*30-$g02h*$G02S-$Comp11H*$Comp11S)/$G03S" | bc -l)`
    sed -e 's/#G02H#/'$g02h'/' -e 's/#G03H#/'$g03h'/' $inconfig > $outconfig
    for i in `seq $reps` ; do
	suffix=`printf %03d $i`
	python3 run_simulation.py -O $outdir -S $suffix $outconfig &
    done
    wait
done

