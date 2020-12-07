#!/bin/bash

output_dir="npz_files"

python3 build_data.py -N 100  -H 2.333333:10.333333 -s test_set -O ${output_dir} > test_data.log
python3 build_data.py -N 100  -H 2.666666:10.666666 -s validation_set -O ${output_dir} > validation_data.log
python3 build_data.py -N 1000 -H 2:10 -s training_set -O ${output_dir} > training_data.log

