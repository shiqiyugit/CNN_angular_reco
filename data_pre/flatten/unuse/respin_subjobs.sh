#!/bin/bash

FLV="Mu"
NUM=14
BIN_MAX=30000
#BIN_MAX=6500
CUTS="CC"
#DIR="/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/level6_hdf5_oldgcd/hdf5/15th/"
DIR="/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/IC19start_allend/"
jobind=16
OUTDIR="/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/IC19start_allend/respin/"
#/mnt/scratch/yushiqi2/flattening/300G/IC19start_allend/"

#for dir in 16k_1st 16k_2nd 1st 2nd 3rd 4th 5th 6th 7th 8th;
#do
#  INDIR=$DIR$dir/
#dir=IC
#for((IND=0;IND<=99;IND+=1));
dir=respin
for IND in 0 1 2 3 4 5 6 7 8 9;
  do
    FILENUM=`printf "%01d\n" $IND`
    infile=NuMu_genie_149999_${FILENUM}_*_level6.zst_cleanedpulses_transformed_IC19lt003_CC_contained_IC19_start_all_end_flat_101bins_50000evtperbin.hdf5
    name=Nu${FLV}_genie_${NUM}9999_${FILENUM}_${dir}_level6.zst_cleanedpulses_transformed_IC19
    
    sed -e "s|@@infile@@|${infile}|g" \
        -e "s|@@bin_max@@|$BIN_MAX|g" \
        -e "s|@@name@@|${name}|g" \
        -e "s|@@cuts@@|${CUTS}|g" \
        -e "s|@@dir@@|${DIR}|g"\
        -e "s|@@outdir@@|${OUTDIR}|g"\
        < job_template_flatten_subset.sb > slurm/flatten_Nu${FLV}_${FILENUM}_${dir}.sb
  done
