
import numpy as np
import glob
import h5py
import argparse
from utils import handle_data
from handle_data import CutMask
from handle_data import VertexMask
import math

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",default='NuMu_genie_149999_029997_level6.zst_cleanedpulses_transformed_IC19.hdf5',
                    type=str,dest="input_file", help="names for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/level6_hdf5_oldgcd/hdf5/7/',
                    dest="path", help="path to input files")
parser.add_argument("-od", "--out_path",type=str,default='./',
                    dest="out_path", help="path to output files")
parser.add_argument("-o", "--output",type=str,default='flat_zenith',
                    dest="output", help="names for output files")

args = parser.parse_args()

file_name = args.path + args.input_file

f = h5py.File(file_name, "r")
#'features_DC', 'features_IC', 'initial_stats', 'input_transform_factors', 'labels', 'num_pulses_per_dom', 'output_label_names', 'output_transform_factors', 'trigger_times', 'weights'
file_features_DC = f["features_DC"][:]
file_features_IC = f["features_IC"][:]
file_labels = f["labels"][:]
file_stats = f["initial_stats"][:]
file_pulses_per_dom = f["num_pulses_per_dom"][:]
try:
    file_trig_times = f["trigger_times"][:]
except:
    file_trig_times = None
    
try:
    file_weights = f["weights"][:]
except:
    file_weights = None
try:
    file_input_transform = f["input_transform_factors"][:]
except:
    file_input_transform = None
try:
    file_output_transform = f["output_transform_factors"][:]
except:
    file_output_transform = None
try:
    file_output_names = f["output_label_names"][:]
except:
    file_output_names = None

f.close()

zenith=file_labels[:,1]
assert np.logical_and(np.max(zenith<3.15), np.min(zenith>=0)),"sanity check on zenith"
uvalues, keep=np.unique(zenith,return_index=True)

print("before uniqued: ", zenith.shape, "after uniqued: ", keep.shape)
