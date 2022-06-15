#!/bin/bash

# this script changes the values of inertia of G4 or G8 while keeping the momentum of area 1 fixed

if [ ! $# -eq 2 ] ; then
    echo "usage: `basename $0` momentum generator"
    echo "        momentum should be either 'low' or 'high'"
    echo "        generator should be either 'G4' or 'G8'"
    exit 1
fi

mom=$1
gen=$2

if [ "$mom" != "low" ] && [ "$mom" != "high" ] ; then
    echo "momentum should be either 'low' or 'high'."
    exit 2
fi

if [ "$gen" != "G4" ] && [ "$gen" != "G8" ] ; then
    echo "generator should be either 'G4' or 'G8'."
    exit 3
fi

N=100
outdir="data/IEEE39/converted_from_PowerFactory/all_stoch_loads/${mom}_momentum_test_var_${gen}"
# default values of inertia of G4 and G8
G04H=3.57
G08H=3.47
# default values of inertia of G2 and G3
if [ "$mom" = "low" ] ; then
    G02H=2.932
    G03H=3.247
else
    G02H=6.146
    G03H=6.059
fi

inconfig="IEEE39_PF_sim_config.json"
outconfig="${inconfig%.json}_${mom}_${gen}.json"

echo "Output directory: $outdir"
mkdir -p $outdir

Hlow=2.5
Hhigh=4.5
H=( $(python3 linspace.py $Hlow $Hhigh $N) )
for h in ${H[@]} ; do
    if [ "$gen" = "G4" ] ; then
	sed -e 's/#G02H#/'$G02H'/' -e 's/#G03H#/'$G03H'/' \
	    -e 's/#G04H#/'$h'/'    -e 's/#G08H#/'$G08H'/' \
	    $inconfig > $outconfig
    else
	sed -e 's/#G02H#/'$G02H'/' -e 's/#G03H#/'$G03H'/' \
	    -e 's/#G04H#/'$G04H'/' -e 's/#G08H#/'$h'/' \
	    $inconfig > $outconfig
    fi
    python3 run_simulation.py -O $outdir $outconfig
done

