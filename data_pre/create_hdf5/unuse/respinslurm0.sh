#!/bin/bash

INDEX_START=1
INDEX_END=23000
dir=L55Sim1000/slurm0done/
slurm=L55Sim1000/slurm0

file_dir=/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/level6_hdf5_oldgcd/hdf5_pulses/1st/
for ((INDEX=${INDEX_START};INDEX<=${INDEX_END};INDEX+=1)); do
  FILE_NR=`printf "%06d" $INDEX`
  job=$dir'NuMu_genie_149999_'$FILE_NR'_level6.zst.sb'
  file=$file_dir'NuMu_genie_149999_'$FILE_NR'_level6.zst_cleanedpulses_transformed_IC78.hdf5'
  if test -f "$job"; then
    if test -f "$file"; then
      echo "$job completed"
    else
      mv $job $slurm
    fi
  fi
done
