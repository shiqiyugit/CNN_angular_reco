#!/bin/bash

FLV="Mu"
NUM=14
#BIN_MAX=7750
#BIN_MAX=6960
#BIN_MAX=6280 #20000
BIN_MAX=999999
CUTS="CC"
DIR="/mnt/scratch/yushiqi2/official/"
OUTDIR="/mnt/scratch/yushiqi2/official/"
#DIR="/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/IC19start_allend/"
#OUTDIR="/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/IC19start_allend/final_right_oscwei/"
#OUTDIR="/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/allstartend/final_right_oscwei/"
#DIR="/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/allstartend/"
for IND in 0 1 2 3 4 5 6 7 8 9;
  do
    FILENUM=`printf "%01d\n" $IND`
#    infile=NuMu_genie_149999_${FILENUM}_${FILENUM}_level6.zst_cleanedpulses_transformed_IC19lt003_CC_all_start_all_end_flat_101bins_30000evtperbin.hdf5
#    infile=NuMu_genie_149999_${FILENUM}_${FILENUM}_level6.zst_cleanedpulses_transformed_IC19lt003_CC_contained_IC19_start_all_end_flat_101bins_30000evtperbin.hdf5
    infile=oscNext_genie_level6.5_v02.00_pass2.140000.0????${IND}.i3.bz2_cleanedpulses_transformed_IC19.hdf5
    name=flatten_${FILENUM}_official_Nu${FLV}_genie_${NUM}0000_level6.5zst_cleanedpulses_transformed_IC19
    
    sed -e "s|@@infile@@|${infile}|g" \
        -e "s|@@bin_max@@|$BIN_MAX|g" \
        -e "s|@@name@@|${name}|g" \
        -e "s|@@cuts@@|${CUTS}|g" \
        -e "s|@@dir@@|${DIR}|g"\
        -e "s|@@outdir@@|${OUTDIR}|g"\
        < job_template_final.sb > official/Nu${FLV}_${FILENUM}.sb
  done
#        < job_template_final.sb > slurmcontained/flatten_Nu${FLV}_${FILENUM}.sb

