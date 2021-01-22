#!/bin/bash

function usage {
    echo "usage: ${progname} [-o output_dir] logfile"
}

progname=`basename $0`
infile=""
verbose="no"

while getopts ":hvo:" opt ; do
    case ${opt} in
	h )
	    usage
	    exit 0
	    ;;
	o )
	    outdir=$OPTARG
	    ;;
	v )
	    verbose="yes"
	    ;;
	: )
	    echo "Invalid option: $OPTARG requires an argument"
	    exit 1
	    ;;
	* )
	    echo "Unknown option: $OPTARG"
            usage
            exit 2
            ;; 
    esac
done
shift $((OPTIND -1))

if [ $# -lt 1 ] ; then
    usage
    exit 3
fi

infile="$1"
if [ ! -f $infile ] ; then
    echo "${infile}: no such file."
    exit 4
fi

if [ -z $outdir ] ; then
    outdir=`dirname $infile`
fi

if [ ! -d $outdir ] ; then
    echo "$outdir: no such directory."
    exit 5
fi

indir=`dirname $infile`
outfile="$outdir/`basename $infile`"

if [ "$verbose" = "yes" ] ; then
    echo " Input file: $indir/$infile"
    echo "Output file: $outfile"
    if [ $outdir = $indir ] ; then
	echo "Backup file: $infile.bak"
    fi
fi

tmpfile="`tempfile`"

cat $infile | tr -d '\b\r' > $tmpfile
sed -i '/^  1\//d' $tmpfile
if [ $outdir = $indir ] ; then
    mv $infile $infile.bak
fi
mv $tmpfile $outfile
bzip2 $outfile

