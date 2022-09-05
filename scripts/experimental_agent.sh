#!/bin/bash

NUM_JOBS=8  # number of jobs to run in parallel, may need to reduce to satisfy computational constraints

# declare an array
data=()
idx=0 

for seed in $(seq 100 101); do
      sem -j${NUM_JOBS} --line-buffer data[idx]=`python -u ABC_experiment_3.py -s ${seed}`
      idx=$((idx+1))
done
sem --wait

echo ${data[*]}
