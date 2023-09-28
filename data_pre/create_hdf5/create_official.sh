#!/bin/bash
#11st  12nd  13rd  14th  15th  16th  
#1st  2nd  3rd  4th  5th  6th  7th  8th
#/mnt/research/IceCube/jmicallef/official_oscnext/level6/1?0000/oscNext_genie_level6.5_v02.00_pass2.1?0000.??????.i3.bz2

INPUTFILES=/mnt/research/IceCube/jmicallef/official_oscnext/level6/140000/oscNext_genie_level6.5_v02.00_pass2.140000.??????.i3.bz2
OUTPUT_DIR=/mnt/scratch/yushiqi2/official/
#research/IceCube/yushiqi2/Files/Flat_Zenith/level6_hdf5_official/hdf5/
LOG_FOLDER=/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/data_pre/create_hdf5/L55Sim1000

[ ! -d $OUTPUT_DIR ] && mkdir $OUTPUT_DIR

[ ! -d $LOG_FOLDER/log ] && mkdir $LOG_FOLDER/log
[ ! -d $LOG_FOLDER/off ] && mkdir $LOG_FOLDER/off

#rm $LOG_FOLDER/slurm0/*
#rm $LOG_FOLDER/log/*
rm $LOG_FOLDER/off/*
for file in $INPUTFILES;
do
    name=`basename $file`

    sed -e "s|@@file@@|${file}|g" \
        -e "s|@@OUTPUT_DIR@@|${OUTPUT_DIR}|g" \
        < case_template_single_file.sb> $LOG_FOLDER/off/${name}.sb 

done

