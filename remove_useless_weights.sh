#!/bin/bash

folder="."
progname="`basename $0`"

if [ $# -eq 1 ] ; then
    if [ "$1" == "-h" ] ; then
	echo "usage: $progname [directory]"
	exit 0
    fi

    if [ -d $1 ] ; then
	folder=$1
    else
	echo "$progname: $1: no such directory"
	exit 1
    fi
fi

h5files=(`find $folder -maxdepth 2 -name weights*-*.h5`)
if [ ${#h5files[@]} -eq 0 ] ; then
    echo "No H5 files to remove in '$folder'."
    exit 0
fi

minloss=100
maxiter=0
for path in ${h5files[@]} ; do
    h5file=`basename $path`
    tmp="${h5file%.h5}"
    loss="${tmp#weights*-}"
    tmp="${h5file%-*.h5}"
    iter="${tmp#weights.}"
    if (( $(echo "$loss <= $minloss" | bc -l) && $(echo "$iter > $maxiter" | bc -l) )); then
	minloss=$loss
	maxiter=$iter
	minfile=$path
    fi
done
d=`dirname $minfile`
echo "`dirname $folder`: moving `basename $minfile` to weights.h5"
mv $minfile $d/tmp
rm -f $d/weights*-*.h5
mv $d/tmp $d/weights.h5
