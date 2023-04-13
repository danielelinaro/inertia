#!/bin/bash

cnt=0
nbands=20
for d in `find experiments/neural_network -maxdepth 1 -mindepth 1 -type d -mtime -1` ; do
    expt=`basename $d`
    python3 compute_IO_correlations.py --plots --nbands $nbands --order 6 $expt &
    let cnt=cnt+1
    if [ $cnt -eq 10 ] ; then
	cnt=0
	wait
    fi
done
wait
