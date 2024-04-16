#!/bin/bash

#for SYST_SET in 120001 120002 120003 120004 120100 120101 120102 120150;
#for SYST_SET in 121118 141118; #140501 140502 140503 140500;
for SYST_SET in 160000;
do 
    SYST=$SYST_SET
    FILEDIR="/mnt/research/IceCube/jmicallef/official_oscnext/level6/${SYST}/"
    OUTDIR="/mnt/research/IceCube/yushiqi2/Files/Zenith_i3_output/official_L6.5/${SYST}/"
    FILENAME="oscNext_genie_level6_v02.00_pass2.${SYST}.*.i3.zst"
    FILES=$FILEDIR$FILENAME
    ind=1
    for f in $FILES;
    do 
        printf -v INDEX "%06d" $ind
        INFILE=`basename $f`
        OUTFILE="${INFILE/level6/level6.5}"
#        OUTFILE=oscNext_genie_level6.5_v02.00_pass2.${SYST}.${INDEX}.i3.bz2
        sed -e "s|@@infile@@|${INFILE}|g" \
            -e "s|@@outfile@@|${OUTFILE}|g" \
            -e "s|@@indir@@|${FILEDIR}|g" \
            -e "s|@@outdir@@|${OUTDIR}|g"\
            < job_template_finalize_retro.sb > slurm/finalize_retro_${SYST}_${INDEX}.sb
        let ind=$ind+1
    done

done
