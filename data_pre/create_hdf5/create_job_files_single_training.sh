#!/bin/bash
#11st  12nd  13rd  14th  15th  16th  
#1st  2nd  3rd  4th  5th  6th  7th  8th

INPUTFILES=/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/level6_oldgcd/$1/NuMu_genie_149999_0?????_level6.zst
OUTPUT_DIR=/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/level6_hdf5_oldgcd/hdf5_allIC/$1/
LOG_FOLDER=/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/data_pre/create_hdf5/L55Sim1000

[ ! -d $OUTPUT_DIR ] && mkdir $OUTPUT_DIR

[ ! -d $LOG_FOLDER/log ] && mkdir $LOG_FOLDER/log
[ ! -d $LOG_FOLDER/slurm0 ] && mkdir $LOG_FOLDER/slurm0

#rm $LOG_FOLDER/slurm0/*
#rm $LOG_FOLDER/log/*
#rm $LOG_FOLDER/slurm1/*

for file in $INPUTFILES;
do
    name=`basename $file`

#respin files
#    hdf5name=${name}_cleanedpulses_transformed_IC19.hdf5
#    if [[ -f $OUTPUT_DIR$hdf5name ]];then
#      continue
#    else
#      echo $hdf5name
#    fi

    sed -e "s|@@file@@|${file}|g" \
        -e "s|@@log@@|${LOG_FOLDER}/log/$name.log|g" \
        -e "s|@@name@@|${name}|g" \
        -e "s|@@OUTPUT_DIR@@|${OUTPUT_DIR}|g" \
        < job_template_single_file.sb> $LOG_FOLDER/slurm0/${name}.sb 

done

