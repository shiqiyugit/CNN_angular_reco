#!/bin/bash --login
INPUT="flat_zenith_E0to3_all_all_start_all_end_flat_101bins_10evtperbin_file0?_contained.hdf5"
INDIR="/data/user/shiqiyu/CNN_angular_reco/data_pre/example/"

OUTDIR="/data/user/shiqiyu/CNN_angular_reco/data_pre/example/"
NUMVAR=1
LR_EPOCH=10 #120
LR_DROP=0.8 #0.55
LR=0.001
OUTNAME="test"

SUB=0.2
DO=0.2
NET="make_network"
START=0
END=100
STEP=10

for ((EPOCH=$START;EPOCH<=$END;EPOCH+=$STEP))
do

  singularity exec -B /data/:/data/ --nv /data/ana/LE/oscNext/flercnn_sample_analysis/flercnn_reco/icetray_stable-tensorflow.sif python /data/user/shiqiyu/CNN_angular_reco/CNN_Train_ic.py --input_files $INPUT --path $INDIR --output_dir $OUTDIR --name $OUTNAME -e $STEP --variables $NUMVAR --no_test True --first_variable zenith --lr $LR --lr_epoch $LR_EPOCH --lr_drop $LR_DROP --sub_dropout $SUB --dropout $DO --network $NET --auto_train True 

done
