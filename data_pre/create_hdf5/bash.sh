#!/bin/bash

for dir in {1..9};do
  echo $dir 
#  sh create_cases.sh $dir
  sbatch run_cases.sh $dir
#  sh create_job_files_single_training.sh $dir
done

#dir=L55Sim1000/slurmdone/
#files=`ls $dir`
#for file in $files;do
#  mv L55Sim1000/slurm0/$file L55Sim1000/sh/
#done
