#!/bin/bash

FILEPATH=/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/data_pre/official
LOG_FOLDER=$FILEPATH/log
JOB_FOLDER=$FILEPATH/slurm

#INDIR=/mnt/scratch/yushiqi2/L6_muongen/
#OUTDIR=/mnt/scratch/yushiqi2/L7_muongun/
INDIR=/mnt/scratch/yushiqi2/L6_nu/
OUTDIR=/mnt/scratch/yushiqi2/L7_nu/

#/mnt/research/IceCube/yushiqi2/Files/Zenith_i3_output/FLERCNN/
#/mnt/scratch/yushiqi2/L6_nu/oscNext_genie_level6.5_v02.00_pass2.120000.000001_FLERCNN.i3.zst
[ ! -d $output ] && mkdir $output
[ ! -d $LOG_FOLDER ] && mkdir $LOG_FOLDER
rm $LOG_FOLDER/*
rm $JOB_FOLDER/*
#nue=644 #muon=19998 numu=1549 nutau=349
FLV=12
NUM=644
DATA=genie

#for file in $INDFILE
for ((i=1;i<=$NUM;i++))
do 
      ind=`printf "%06d" $i`
      echo $ind
      outname=oscNext_${DATA}_level7_v02.00_pass2.${FLV}0000.${ind}
      inname=oscNext_${DATA}_level6_v02.00_pass2.${FLV}0000.${ind}.i3.zst
#oscNext_genie_level6.5_v02.00_pass2.${FLV}0000.${ind}_FLERCNN.i3.zst
#      name=`basename $file`
      sed -e "s|@@indir@@|${INDIR}|g"\
          -e "s|@@outdir@@|${OUTDIR}|g"\
          -e "s|@@outfile@@|${outname}|g"\
          -e "s|@@infile@@|${inname}|g"\
          -e "s|@@log@@|${LOG_FOLDER}/${ind}.log|g" \
          < job_template_final_retro_CNN.sb > $JOB_FOLDER/${ind}.sb
done
#cp run_all_here.sh $JOB_FOLDER
#          -e "s|@@infile@@|${inname}|g"\
#          -e "s|@@outfile@@|${outname}|g"\
#          -e "s|@@indir@@|${INDIR}|g"\
#          -e "s|@@vars@@|${VARS}|g"\
