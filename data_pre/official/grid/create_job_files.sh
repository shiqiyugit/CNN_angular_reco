#!/bin/bash

FILEPATH=/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/data_pre/official/grid
LOG_FOLDER=$FILEPATH/logs
JOB_FOLDER=$FILEPATH/slurm_numu

MODEL_DIR="/mnt/research/IceCube/FLERCNN/"
MODEL_NAME="energy_FLERCNN.hdf5,zenith_FLERCNN.hdf5,XYZ_FLERCNN.hdf5,PID_FLERCNN.hdf5"
VARS="energy,zenith,vertex_x,vertex_y,vertex_z,prob_track"
#output=/mnt/research/IceCube/yushiqi2/Files/Zenith_i3_output/FLERCNN/

FLV=14 #12 #2346
DATA=genie #muongen
TMPDIR=/mnt/scratch/yushiqi2/L6_nu/
NUM=1549 #nue=644 #muon=19998 numu=1549 nutau=349

[ ! -d $output ] && mkdir $output
[ ! -d $LOG_FOLDER ] && mkdir $LOG_FOLDER

for ((i=1;i<=$NUM;i++))
do 
      ind=`printf "%06d" $i`
      echo $ind
#      name=`basename $file`
      sed -e "s|@@fileind@@|${ind}|g" \
          -e "s|@@log@@|${LOG_FOLDER}/${ind}.log|g" \
          -e "s|@@flvind@@|${FLV}|g" \
          -e "s|@@dataset@@|${DATA}|g" \
          -e "s|@@dir@@|${TMPDIR}|g" \
          < job_template.sb > $JOB_FOLDER/${ind}.sb
done
#cp run_all_here.sh $JOB_FOLDER
#          -e "s|@@outdir@@|${output}|g"\
#          -e "s|@@models@@|${MODEL_NAME}|g"\
#          -e "s|@@model_dir@@|${MODEL_DIR}|g"\
#          -e "s|@@vars@@|${VARS}|g"\
