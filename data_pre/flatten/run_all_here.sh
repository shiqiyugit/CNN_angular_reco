#!/bin/bash

#FILES="slurmcontained/flatten_NuMu_?.sb "
#FILES="slurmcontained/*.sb"
#FILES="slurm/*.sb"
#FILES="slurm/flatten_NuMu_?.sb "
FILES="official/NuMu_*.sb"
#donedir="slurm/done/"
for f in $FILES
do
    sbatch $f
#    cp $f $donedir
done
