#!/bin/bash

#FILES=L55Sim1000/off/*.sb
FILES=flatICs/*.sb

njobs=`squeue -u yushiqi2 -h -t pending,running -r | wc -l`
let space=1000-$njobs

for f in $FILES
do
    if [[ $space -gt 2 ]];then
      sbatch $f
      cp $f job_logs/
      let space=$space-1
    fi
    
    while (( $space <=2 ))
    do
      njobs=`squeue -u yushiqi2 -h -t pending,running -r | wc -l`
      if [[ $njobs -gt 997 ]];then
        echo $njobs "running, sleeping for another 3m" 
        sleep 3m
      else
        echo $njobs "running, keep submitting jobs..."
        let space=1000-$njobs
      fi
    done

done
