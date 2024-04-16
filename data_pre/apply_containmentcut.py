import numpy as np
import h5py
import os, sys

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",default='flat_zenith_E0to3_all_all_start_all_end_flat_101bins_10evtperbin_file01.hdf5', 
                    type=str,dest="input_file", help="name for ONE input file")
parser.add_argument("-d", "--path",type=str,default='example/', 
                    dest="path", help="path to input files")
parser.add_argument("--tmax",type=float,default=200.0,
                    dest="tmax",help="Multiplication factor for track length")
parser.add_argument("--start",type=str,default="contained_start",
                    dest="start",help="cut of start point")
parser.add_argument("--end",type=str,default="contained_end",
                    dest="end",help="cut of end point")
args = parser.parse_args()

input_file = args.path + args.input_file
track_max = args.tmax

f = h5py.File(input_file, 'r')
Y_test = f['Y_test'][:]
X_test_DC = f['X_test_DC'][:]
X_test_IC = f['X_test_IC'][:]

Y_train = f['Y_train'][:]
X_train_DC = f['X_train_DC'][:]
X_train_IC = f['X_train_IC'][:]

Y_validate = f['Y_validate'][:]
X_validate_DC = f['X_validate_DC'][:]
X_validate_IC = f['X_validate_IC'][:]

try:
    reco_test = f['reco_test'][:]
    reco_train = f['reco_train'][:]
    reco_validate = f['reco_validate'][:]
except:
    reco_test = None
    reco_train = None
    reco_validate = None
f.close()
del f
print("Test: %i, Train: %i, Validate: %i"%(Y_test.shape[0],Y_train.shape[0],Y_validate.shape[0]))
if reco_test is not None:
    print("Recontruction included in file")
    print(reco_test.shape,reco_train.shape,reco_validate.shape)
start_cut=args.start
end_cut=args.end
# Apply Cuts
from utils.handle_data import VertexMask
test_mask = VertexMask(Y_test,azimuth_index=7,track_index=2,max_track=track_max)
vtx_test=np.logical_and(test_mask[start_cut],test_mask[end_cut])

Y_test = Y_test[vtx_test]
X_test_DC = X_test_DC[vtx_test]
X_test_IC = X_test_IC[vtx_test]
if reco_test is not None:
    reco_test = reco_test[vtx_test]

train_mask = VertexMask(Y_train,azimuth_index=7,track_index=2,max_track=track_max)
vtx_train=np.logical_and(train_mask[start_cut],train_mask[end_cut])

Y_train = Y_train[vtx_train]
X_train_DC = X_train_DC[vtx_train]
X_train_IC = X_train_IC[vtx_train]
if reco_train is not None:
    reco_train = reco_train[vtx_train]

val_mask = VertexMask(Y_validate,azimuth_index=7,track_index=2,max_track=track_max)
vtx_validate=np.logical_and(val_mask[start_cut],val_mask[end_cut])

Y_validate = Y_validate[vtx_validate]
X_validate_DC = X_validate_DC[vtx_validate]
X_validate_IC = X_validate_IC[vtx_validate]
if reco_validate is not None:
    reco_validate = reco_validate[vtx_validate]
print(Y_test.shape[0],Y_train.shape[0],Y_validate.shape[0])


output_file = input_file[:-5] + "_contained.hdf5"
print("Putting new file at %s"%output_file)

f = h5py.File(output_file, "w")
f.create_dataset("Y_test", data=Y_test)
f.create_dataset("X_test_DC", data=X_test_DC)
f.create_dataset("X_test_IC", data=X_test_IC)
f.create_dataset("Y_train", data=Y_train)
f.create_dataset("X_train_DC", data=X_train_DC)
f.create_dataset("X_train_IC", data=X_train_IC)
f.create_dataset("Y_validate", data=Y_validate)
f.create_dataset("X_validate_DC", data=X_validate_DC)
f.create_dataset("X_validate_IC", data=X_validate_IC)
if reco_test is not None:
    f.create_dataset("reco_test", data=reco_test)
    f.create_dataset("reco_train", data=reco_train)
    f.create_dataset("reco_validate", data=reco_validate)
f.close()
