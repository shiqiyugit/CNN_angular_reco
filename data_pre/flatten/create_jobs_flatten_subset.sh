#!/bin/bash

FLV="Mu"
NUM=14
BIN_MAX=30000
CUTS="CC"
DIR=/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/level6_hdf5_oldgcd/hdf5/
#jobind=16
#OUTDIR="/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/allstartend/"
OUTDIR=/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/IC19start_allend/

for dir in {0..9}; #11st  12nd  13rd  14th  15th  16th  1st  2nd  3rd  4th  5th  6th  7th  8th;
do
    INDIR=$DIR$dir/
#  [ ! -d $INDIR ] && mkdir $INDIR
#  for IND in 0 1 2 3 4 5 6 7 8 9;
#  do
    FILENUM=`printf "%01d\n" $dir`
    infile=Nu${FLV}_genie_${NUM}9999_0????${FILENUM}_level6.zst_cleanedpulses_transformed_IC19.hdf5
    name=Nu${FLV}_genie_${NUM}9999_${FILENUM}_${dir}_level6.zst_cleanedpulses_transformed_IC19
    
    sed -e "s|@@infile@@|${infile}|g" \
        -e "s|@@bin_max@@|$BIN_MAX|g" \
        -e "s|@@name@@|${name}|g" \
        -e "s|@@cuts@@|${CUTS}|g" \
        -e "s|@@dir@@|${INDIR}|g"\
        -e "s|@@outdir@@|${OUTDIR}|g"\
        < job_template_flatten_subset.sb > slurmcontained/flatten_Nu${FLV}_${FILENUM}_${dir}.sb
#  done
done
