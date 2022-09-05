#!/bin/bash

NUM_JOBS=32  # number of jobs to run in parallel, may need to reduce to satisfy computational constraints

config=${config:-rmsc03}
ticker=${ticker:-ABM}
date=${date:-20200603}
seed=${seed:-1234}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
   fi

  shift
done

cd ../project/abides && python -u abides.py -c $config -t $ticker -d $date -s $seed -l $loc && cd ../..
