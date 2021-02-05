#!/bin/bash

function usage {
    echo "usage: ${progname} [-Hmin hmin] [-Hmax hmax] [-G gen_id] [-D damping] [-DZA dead-band width] [-f | --force] pan_file"
}

progname=`basename $0`
D="2"
DZA="60.0"
Hmin="2.0"
Hmax="10.0"
generator="1"
force="no"

while [[ $# -gt 0 ]] ; do
    key="$1"
    case $key in
	-D)
	    D="$2"
	    shift
	    shift
	    ;;
	-DZA)
	    DZA="$2"
	    shift
	    shift
	    ;;
	-G)
	    generator="$2"
	    shift
	    shift
	    ;;
	-f|--force)
	    force="yes"
	    shift
	    ;;
	-h|--help)
	    usage
	    exit 0
	    ;;
	-Hmin)
	    Hmin="$2"
	    shift
	    shift
	    ;;
	-Hmax)
	    Hmax="$2"
	    shift
	    shift
	    ;;
	*)
	    if [ -f "${key}" ] ; then
		panfile="${key}"
	    else
		echo "unknown option: ${key}"
		usage
		exit 1
	    fi
	    shift
	    ;;
    esac
done

if [ -z "$panfile" ] ; then
    usage
    exit 1
fi

if [ "$(basename $panfile)" = "ieee14.pan" ] ; then
    prefix="IEEE14"
elif [ "$(basename $panfile)" = "two-area.pan" ] ; then
    prefix="TWO_AREA"
else
    prefix=""
fi

output_dir="in_progress/${prefix}_D=${D}_DZA=${DZA}"
config_template="config/build_data_config_template_${prefix}.json"
training_config=$(mktemp --suffix "_training_config.json")
test_config=$(mktemp --suffix "_test_config.json")
validation_config=$(mktemp --suffix "_validation_config.json")

if [ -d $output_dir ] && [ "$force" = "no" ] ; then
    echo "Directory ${output_dir} exists: use -f to force (potential) overwrite of files."
    exit 1
fi

sed -e 's/{HMIN}/'$Hmin'/' -e 's/{HMAX}/'$Hmax'/' -e 's/{D}/'$D'/' -e 's/{GEN_ID}/'$generator'/' \
    -e 's/{DZA}/'$DZA'/' -e 's/{N}/1000/' ${config_template} > ${training_config}

Hmin=`echo $Hmin+0.333333 | bc`
Hmax=`echo $Hmax+0.333333 | bc`
sed -e 's/{HMIN}/'$Hmin'/' -e 's/{HMAX}/'$Hmax'/' -e 's/{D}/'$D'/' -e 's/{GEN_ID}/'$generator'/' \
    -e 's/{DZA}/'$DZA'/' -e 's/{N}/100/' ${config_template} > ${test_config}

Hmin=`echo $Hmin+0.333333 | bc`
Hmax=`echo $Hmax+0.333333 | bc`
sed -e 's/{HMIN}/'$Hmin'/' -e 's/{HMAX}/'$Hmax'/' -e 's/{D}/'$D'/' -e 's/{GEN_ID}/'$generator'/' \
    -e 's/{DZA}/'$DZA'/' -e 's/{N}/100/' ${config_template} > ${validation_config}

python3 build_data.py -s training_set -o ${output_dir} ${training_config} > training_data.log
python3 build_data.py -s test_set -o ${output_dir} ${test_config} > test_data.log
python3 build_data.py -s validation_set -o ${output_dir} ${validation_config} > validation_data.log

mv ${training_config} ${output_dir}
mv ${test_config} ${output_dir}
mv ${validation_config} ${output_dir}

