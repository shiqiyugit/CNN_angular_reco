#!/bin/bash

INDEX_START=1
INDEX_END=23000
dir=L55Sim1000/slurm0done
slurm=L55Sim1000/slurm0
cd $slurm 

files=`ls *.sb`
cd ../..

for file in $files;do
  echo $file
  if test -f $dir/$file; then
   mv $slurm/$file $dir
  fi
done
  
