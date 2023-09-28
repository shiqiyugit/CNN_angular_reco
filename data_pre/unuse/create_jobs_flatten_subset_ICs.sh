#!/bin/bash

FLV="Mu"
NUM=14
BIN_MAX=20000
CUTS="CC"
DIR=/mnt/scratch/yushiqi2/hdf5_allIC/
INFILE=NuMu_genie_149999_??????_level6.zst_cleanedpulses_transformed_IC78.hdf5

OUTDIR="/mnt/scratch/yushiqi2/flattening/300G/AllIC/splitted/"

dir=IC
for((IND=0;IND<=99;IND+=1));
do
    FILENUM=`printf "%02d\n" $IND`
    infile=NuMu_genie_149999_????${FILENUM}_level6.zst_cleanedpulses_transformed_IC78.hdf5
#NuMu_genie_149999_??${FILENUM}_IC_level6.zst_cleanedpulses_transformed_IC78lt003_CC_all_start_all_end_flat_101bins_71evtperbin.hdf5
#Nu${FLV}_genie_${NUM}9999_0??${FILENUM}_level6.zst_cleanedpulses_transformed_IC78.hdf5
#NuMu_genie_149999_?${FILENUM}_IC_level6.zst_cleanedpulses_transformed_IC78lt003_CC_all_start_all_end_flat_101bins_20evtperbin.hdf5
    name=Nu${FLV}_genie_${NUM}9999_${FILENUM}_${dir}_level6.zst_cleanedpulses_transformed_IC78
    
    sed -e "s|@@infile@@|${infile}|g" \
        -e "s|@@bin_max@@|$BIN_MAX|g" \
        -e "s|@@name@@|${name}|g" \
        -e "s|@@cuts@@|${CUTS}|g" \
        -e "s|@@dir@@|${DIR}|g"\
        -e "s|@@outdir@@|${OUTDIR}|g"\
        < job_template_flatten_subset_IC.sb > ICs/flatten_Nu${FLV}_${FILENUM}_${dir}.sb
  done
