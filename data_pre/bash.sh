#!/bin/bash

for ((i=0;i<=8;i+=1));do
  python apply_containmentcut.py -i flatten_NuMu_genie_149999_E0to3_CC_all_start_all_end_flat_101bins_36170evtperbin_file0${i}.hdf5
#  mv /mnt/scratch/yushiqi2/training_files/L55Sim1000/flatten_NuMu_genie_149999_E0to3_CC_all_start_all_end_flat_101bins_20000evtperbin_file0${i}.hdf5 /mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/level6_hdf5/NuMu_genie_149999_level6.zst_cleanedpulses_transformed_IC19_CC_all_end_all_end_flat_101bins_20000evtperbin_splitted_file0${i}.hdf5

#  for ((j=0;j<=9;j+=1));do
#    mv /mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/300G/end_IC19/allendtill7th/NuMu_genie_149999_${j}_0${i}_level6.zst_cleanedpulses_transformed_IC19lt003_CC_all_start_all_end_IC19_flat_101bins_5600evtperbin.hdf5 /mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/300G/end_IC19/allendtill7th/NuMu_genie_149999_${j}_0${i}_level6.zst_cleanedpulses_transformed_IC19lt003_CC_all_start_all_end_flat_101bins_5600evtperbin.hdf5
#  done
done
