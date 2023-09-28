#!/bin/bash

FILES="slurm/*_IC.sb"
dir=/mnt/scratch/yushiqi2/flattening/300G/AllIC/

for((ind=0;ind<=999;ind+=1));
do
    FILENUM=`printf "%03d\n" $ind`
    file=NuMu_genie_149999_${FILENUM}_IC_level6.zst_cleanedpulses_transformed_IC78lt003_CC_all_start_all_end_flat_101bins_15000evtperbin.hdf5

    if test -f $dir/$file; then
      continue
#echo $dir/$file
    else
      sb=slurm/flatten_NuMu_${FILENUM}_IC.sb
      #sbatch $sb
      echo "moved" $sb
      mv $sb tmp/
    fi    

done

#for f in $FILES
#do
#    sbatch $f
#done
