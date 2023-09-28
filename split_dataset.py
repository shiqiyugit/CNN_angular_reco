#################################
# Plots input and output features for ONE file
#   Inputs:
#       -i input file:  name of ONE file
#       -d  path:       path to input files
#       -o  outdir:     path to output_plots directory or where final dir will be created
#       -n  name:       Name of directory to create in outdir (associated to filenames)
#       --emax:         Energy max cut, plot events below value, used for UN-TRANSFORM
#       --emin:         Energy min cut, plot events above value
#       --tmax:         Track factor to multiply, use for UN-TRANSFORM
#   Outputs:
#       File with count in each bin
#       Histogram plot with counts in each bin
#################################

import numpy as np
import h5py
import os, sys
import matplotlib
#matplotlib.use('Agg')
#matplotlib.rcParams['agg.path.chunksize'] = 100.

import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as colors
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--number", type=int, default=0, dest="number")
args = parser.parse_args()
ifile=args.number
parser.add_argument("-i", "--input_files",type=str,default='NuMu_genie_149999_'+"{:02d}".format(ifile)+'_level6.zst_cleanedpulses_transformed_IC78_CC_contained_start_contained_end_flat_70evt_500bin_energy_0to5_CC_contained_start_contained_end_flat_500bins_70evtperbin.hdf5',
#flatten_NuMu_genie_149999_level6.zst_cleanedpulses_transformed_IC78_'+str(ifile)+'_10_E0to3_CC_all_start_all_end_flat_101bins_7750evtperbin.hdf5',
                    dest="input_files", help="name and path for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/scratch/yushiqi2/flattening/300G/AllIC/splitted/flat/', 
#/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/AllIC/transformed/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output_dir",type=str,default='/mnt/scratch/yushiqi2/allICs/',
#/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/AllIC/transformed/split/',
                    dest="output_dir", help="path to output_plots directory, do not end in /")

args = parser.parse_args()

input_file = args.path + args.input_files
output_file = args.output_dir+args.input_files

# Here in case you want only one (input can take a few min)
f = h5py.File(input_file, 'r')

keys=list(f.keys())
keys=[key for key in keys if "test" not in key]
print(keys)
train=int(f['Y_train'].shape[0]/10)
valid=int(f['Y_validate'].shape[0]/10)

for key in keys:
  print(key, ": ", f[key].shape[0])

for i in range(0,10):
  filename=args.output_dir+"%d_of_10%d.hdf5"%(i,ifile)
  outf = h5py.File(filename, "w")
  if i == 9:
    for key in keys:
      if key.find("train") != -1:
        lenth=train
      else:
        lenth=valid
      outf.create_dataset(key, data=f[key][i*lenth:-1])
      print(i*lenth, " -- ", f[key].shape[0])

  else:
    for key in keys:
      if key.find("train") != -1:
        lenth=train
      else:
        lenth=valid
      print(i*lenth, " -- ", (i+1)*lenth)

      outf.create_dataset(key, data=f[key][i*lenth:(i+1)*lenth])
  outf.close()

quit()

Y_test = f['Y_test'][:]
#Y_test[:,1]=np.arccos(Y_test[:,1])

X_test_DC = f['X_test_DC'][:]
X_test_IC = f['X_test_IC'][:]
    
Y_train = f['Y_train'][:]
#Y_train[:,1] = np.arccos(Y_train[:,1])

X_train_DC = f['X_train_DC'][:]
X_train_IC = f['X_train_IC'][:]

X_zeros=np.ones([X_train_IC.shape[0], 1, 60, 5])
for i in range(5):
  X_zeros[:,:,:,i]*=np.min(X_train_IC[:,:,:,i])

X_train_IC=np.concatenate((X_train_IC, X_zeros), axis=1)
index1=chunck_blocks(block1)
X_train_IC1 = X_train_IC[:,index1, :,:].reshape((-1,5,5, 60,5))
index2=chunck_blocks(block2)
X_train_IC2 = X_train_IC[:,index2, :,:].reshape((-1,5,5, 60,5))
index3=chunck_blocks(block3)
X_train_IC3 = X_train_IC[:,index3, :,:].reshape((-1,5,6, 60,5))

Y_validate = f['Y_validate'][:]
#Y_validate[:,1] = np.arccos(Y_validate[:,1])

X_validate_DC = f['X_validate_DC'][:]
X_validate_IC = f['X_validate_IC'][:]
    
X_zeros=np.zeros([X_validate_IC.shape[0], 1, 60, 5])
    
X_validate_IC=np.concatenate((X_validate_IC, X_zeros), axis=1)
    
X_validate_IC1 = X_validate_IC[:,index1, :,:].reshape((-1,5,5, 60,5))
    
X_validate_IC2 = X_validate_IC[:,index2, :,:].reshape((-1,5,5, 60,5))
   
X_validate_IC3 = X_validate_IC[:,index3, :,:].reshape((-1,5,6, 60,5))


f.close()
del f

f = h5py.File(output_file, "w")
    
f.create_dataset("Y_train", data=Y_train)
f.create_dataset("Y_test", data=Y_test)
f.create_dataset("X_train_DC", data=X_train_DC)
f.create_dataset("X_test_DC", data=X_test_DC)
f.create_dataset("X_train_IC1", data=X_train_IC1)
f.create_dataset("X_train_IC2", data=X_train_IC2)
f.create_dataset("X_train_IC3", data=X_train_IC3)
f.create_dataset("X_test_IC", data=X_test_IC)
f.create_dataset("Y_validate", data=Y_validate)
f.create_dataset("X_validate_IC1", data=X_validate_IC1)
f.create_dataset("X_validate_IC2", data=X_validate_IC2)
f.create_dataset("X_validate_IC3", data=X_validate_IC3)

f.create_dataset("X_validate_DC", data=X_validate_DC)


f.close()
