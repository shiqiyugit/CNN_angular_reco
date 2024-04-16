#!/bin/bash

FILES="slurm/*.sb"

njobs=`squeue -u yushiqi2 -h -t pending,running -r | wc -l`
let space=1000-$njobs

for f in $FILES
do
    if [[ $space -gt 2 ]];then
      sbatch $f
      let space=$space-1
    fi

    while (( $space <=2 ))
    do
      njobs=`squeue -u yushiqi2 -h -t pending,running -r | wc -l`
      if [[ $njobs -gt 997 ]];then
        echo $njobs "running, sleeping for another 5m" 
        sleep 5m
      else
        echo $njobs "running, keep submitting jobs..."
        let space=1000-$njobs
      fi
    done
done

#echo "start to sleep for 20 min..."
#sleep 1200
#echo "submitting 2nd set of jobs..."

#FILES="slurm1/*.sb"
#for f in $FILES
#do
#    unset I3_BUILD
#    sbatch $f
#done
