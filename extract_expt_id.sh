#/bin/bash
progname=`basename $0`
if [ ! $# -eq 1 ] ; then
    echo "usage: $progname file"
    exit 1
fi
logfile="$1"
if [ ! -f $logfile ] ; then
    echo "$progname: $logfile: no such file."
    exit 2
fi
grep 'Experiment is live on' $logfile | awk '{ print substr($8, length($8)-31, length($8)) }'
