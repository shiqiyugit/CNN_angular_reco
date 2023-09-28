#!/bin/bash

FLV="Mu"
NUM=14
BIN_MAX=70
CUTS="CC"
DIR=/mnt/scratch/yushiqi2/flattening/300G/AllIC/splitted/
INFILE=NuMu_genie_149999_??_IC_level6.zst_cleanedpulses_transformed_IC78lt005_CC_contained_start_contained_end_flat_500bins_20000evtperbin.hdf5

OUTDIR="/mnt/scratch/yushiqi2/flattening/300G/AllIC/splitted/flat/"

for((IND=0;IND<=99;IND+=1));
do
    FILENUM=`printf "%02d\n" $IND`
    infile=NuMu_genie_149999_${FILENUM}_IC_level6.zst_cleanedpulses_transformed_IC78lt005_CC_contained_start_contained_end_flat_500bins_20000evtperbin.hdf5
    name=Nu${FLV}_genie_${NUM}9999_${FILENUM}_level6.zst_cleanedpulses_transformed_IC78_CC_contained_start_contained_end_flat_${BIN_MAX}evt_500bin
    
    sed -e "s|@@infile@@|${infile}|g" \
        -e "s|@@bin_max@@|$BIN_MAX|g" \
        -e "s|@@name@@|${name}|g" \
        -e "s|@@cuts@@|${CUTS}|g" \
        -e "s|@@dir@@|${DIR}|g"\
        -e "s|@@outdir@@|${OUTDIR}|g"\
        < job_template_flatten_subset_IC.sb > flatICs/flatten_Nu${FLV}_${FILENUM}_${dir}.sb
  done
