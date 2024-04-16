#!/bin/bash --login

########### SBATCH Lines for Resource Request ##########

#SBATCH --time=00:59:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                 # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --mem=20G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name i3_test      # you can give your job a name for easier identification (same as -J)
#SBATCH --output @@log@@

########### Command Lines to Run ##################
echo "start"

eval `/cvmfs/icecube.opensciencegrid.org/py2-v3.1.1/setup.sh`

export X509_USER_PROXY=/mnt/home/$USER/x509up_u6014492

#echo $X509_CERT_DIR
flv=12
dataset=genie
#datadir=/data/ana/LE/oscNext/pass2/genie/level6/${flv}0000/
datadir=/data/ana/LE/oscNext/pass2/${dataset}/level7_flercnn/${flv}0000/
infiles=oscNext_genie_level7_flercnn_pass2.${flv}0000.*.i3.zst
TMPDIR=flercnn

mkdir ${TMPDIR}/${flv}0000/

echo "grab file: " time
globus-url-copy -sync gsiftp://gridftp.icecube.wisc.edu${datadir}/$infiles $TMPDIR/$infiles
echo "files done: " time

outfile=oscNext_${dataset}_level6.5_v02.00_pass2.${flv}0000.${ind}
flercnninfile=${outfile}.i3.bz2
flercnnfile=${outfile}_FLERCNN
echo "run L6.5" time
/cvmfs/icecube.opensciencegrid.org/users/Oscillation/software/oscNext_meta/releases/V01-00-05/build__py2-v3.1.1__osgvo-el7/env-shell.sh python /mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/data_pre/official/RunL7.py  -i $infiles -o $outfile -id $TMPDIR -od $TMPDIR
echo "start cnn" time
singularity exec -B /mnt/scratch/yushiqi2:/mnt/scratch/yushiqi2 --nv /mnt/research/IceCube/Software/icetray_stable-tensorflow.sif sh -c "/usr/local/icetray/env-shell.sh python /mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/CNN_Test_multi_i3.py -i $TMPDIR/$flercnninfile -o $TMPDIR/ --name $flercnnfile --cleaned True"

echo "done all" time
#mydir=/data/ana/LE/oscNext_Zenith/FLERCNN_L6/${flv}0000/

#globus-url-copy $TMPDIR/${outfile}.i3.zst gsiftp://gridftp.icecube.wisc.edu$mydir/$outfile

exit $?
