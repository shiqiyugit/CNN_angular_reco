#################################
# Checks number of events in energy bin from hdf5 training data sets before final flattening
#   Inputs:
#       -i input files: name of files (can use * and ?)
#       -d  path:       path to input files
#       -o  outdir:     path to output_plots directory or where final dir will be created
#       -n  name:       Name of directory to create in outdir (associated to filenames)
#       -c  cuts:       name of cuts you want to apply (i.e. track only = track)
#       --emax:         Energy max cut, keep all events below value
#       --emin:         Energy min cut, keep all events above value
#       --tmax:         Track factor to multiply, only used IF TRANFORMED IS TRUE
#       --transformed:  use flag if file has already been transformed
#       --labels:       name of truth array to load (labels, Y_test, Y_train, etc.)
#       --bin_size:     Size (in GeV) for bins to distribute zenith into
#       --start:        Name of vertex start cut
#       --end:          Name of ending position cut
#   Outputs:
#       File with count in each bin
#       Histogram plot with counts in each bin
#################################

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import glob
import os
from utils import handle_data
from handle_data import CutMask
from handle_data import VertexMask
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",default=
#'NuMu_genie_149999_??_IC_level6.zst_cleanedpulses_transformed_IC78_zenith_1to500_CC_all_start_all_end_flat_499bins_20000evtperbin.hdf5',
#NuMu_genie_149999_??_IC_level6.zst_cleanedpulses_transformed_IC78_zenith_1to500_CC_contained_start_contained_end_flat_499bins_20000evtperbin.hdf5',
'NuMu_genie_149999_??_level6.zst_cleanedpulses_transformed_IC78_CC_contained_start_contained_end_flat_70evt_500bin_energy_0to5_CC_contained_start_contained_end_flat_500bins_70evtperbin.hdf5',
#'NuMu_genie_149999_00_IC_level6.zst_cleanedpulses_transformed_IC78lt005_CC_contained_start_contained_end_flat_500bins_20000evtperbin.hdf5',
#'NuMu_genie_149999_0?????_level6.zst_cleanedpulses_transformed_IC78.hdf5',
#'NuMu_genie_149999_level6.zst_cleanedpulses_transformed_IC19_?_10_E0to3_CC_contained_IC19_start_contained_IC19_end_flat_101bins_6285evtperbin.hdf5',
#'NuMu_genie_149999_0?????_level6.zst_cleanedpulses_transformed_IC19.hdf5',
#'NuMu_genie_149999_*_level6.zst_cleanedpulses_transformed_IC19lt003_CC_all_start_all_end_flat_101bins_30000evtperbin.hdf5',
#'NuMu_genie_149999_*_level6.zst_cleanedpulses_transformed_IC19lt003_CC_contained_IC19_start_all_end_flat_101bins_20000evtperbin.hdf5',
#'NuMu_genie_149999_*_level6.zst_cleanedpulses_transformed_IC19lt003_CC_contained_IC19_start_all_end_flat_101bins_50000evtperbin.hdf5',
#'flatten_NuMu_genie_149999_level6.zst_cleanedpulses_transformed_IC19_?_E0to3_CC_contained_IC19_start_all_end_flat_101bins_7777evtperbin.hdf5',
#'NuMu_genie_149999_?_respin_level6.zst_cleanedpulses_transformed_IC19lt003_CC_contained_IC19_start_all_end_flat_101bins_30000evtperbin.hdf5',
#'NuMu_genie_149999_?_IC_level6.zst_cleanedpulses_transformed_IC78_E0to3_CC_all_start_all_end_flat_101bins_6550evtperbin.hdf5',
#'NuMu_genie_149999_0?????_level6.zst_cleanedpulses_transformed_IC78.hdf5',
#'NuMu_genie_149999_??_IC_level6.zst_cleanedpulses_transformed_IC78_E0to3_CC_all_start_all_end_flat_101bins_16evtperbin.hdf5',
                    type=str,dest="input_files", help="names for input files")
parser.add_argument("-d", "--path",type=str,default=
#'/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/level6_hdf5_oldgcd/300G/',
#'/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/allstartend/',
#'/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/IC19start_allend/',
#'/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/level6_hdf5_oldgcd/hdf5/*/',
#'/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/IC19startend/',
#'/mnt/research/IceCube/yushiqi2/Files/Energy/',
#'/mnt/scratch/yushiqi2/hdf5_allIC/',
'/mnt/scratch/yushiqi2/flattening/300G/AllIC/splitted/flat/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/data_pre/',
#/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/',
                    dest="outdir", help="out directory for plots")
parser.add_argument("-n", "--name",type=str,default="zenith_bin_max",
                    dest="name", help="name for output folder")
parser.add_argument("-b", "--bin_size",type=float,default=1, #0.0314,
                    dest="bin_size", help="Size of zenith bins")
parser.add_argument("--emax",type=float,default=500.0,
                    dest="emax",help="Cut anything greater than this energy (in GeV)")
parser.add_argument("--emin",type=float,default=1.0,
                    dest="emin",help="Cut anything less than this energy (in GeV)")
parser.add_argument("--tmax",type=float,default=200.0,
                    dest="tmax",help="Multiplication factor for track")
parser.add_argument("-c", "--cuts",type=str, default="CC",
                    dest="cuts", help="Type of events to keep (all, cascade, track, CC, NC, etc.)")
parser.add_argument("--labels",type=str,default="Y_train",
                    dest="labels", help="name of truth array to read in from input files")
parser.add_argument("--transformed",default=True,action='store_true',
                    dest="transformed", help="add flag if labels truth input is already transformed")
parser.add_argument("-s", "--start",type=str, default="contained_start", #all_start",
                    dest="start_cut", help="Vertex start cut (all_start, old_start_DC, start_DC, start_IC, start_IC19)")
parser.add_argument("-e", "--end",type=str, default="contained_end",
                    dest="end_cut", help="End position cut (end_start, end_IC7, end_IC19)")
args = parser.parse_args()

input_files = args.input_files
files_with_paths = args.path + input_files
event_file_names = sorted(glob.glob(files_with_paths))

emax = args.emax
emin = args.emin
track_max = args.tmax
bin_size = args.bin_size
cut_name = args.cuts
truth_name = args.labels
start_cut = args.start_cut
end_cut = args.end_cut
transformed = args.transformed
efactor=100
zmin=0.
import math
zmax=math.pi
azimuth_index = 2
track_index = 7

zenith_bin_array = np.arange(zmin,zmax,bin_size)

print("Cutting Emax %.f, emin %.f, with event type %s, start cut: %s and end cut: %s"%(emax,emin,cut_name,start_cut,end_cut))
    
if args.name == "None":
    file_name = event_file_names[0].split("/")
    output = file_name[-1][:-5]
else:
    output = args.name
outdir = args.outdir + output
if os.path.isdir(outdir) != True:
    os.mkdir(outdir)

# Find number of bins
bins = 100
#int((emax-emin)/float(bin_size))
if zmax%bin_size !=0:
    bins +=1 #Put remainder into additional bin
count_zenith = np.zeros((bins))
all_energy=[]
all_zenith=[]
all_weights=[]
tot=0
print("Evts in File\t After Type Cut\t After Energy Cut")
for a_file in event_file_names:
    ### Import Files ###
    f = h5py.File(a_file, 'r')
    try:
      file_labels = f[truth_name][:]
      #file_weights=f['weights_train'][:]
      tot+=file_labels.shape[0]
    except:
      f.close()
      del f
      continue
    f.close()
    del f

    if file_labels.shape[0] == 0:
        print("Empty file...skipping...")
        continue

    if transformed:
        azimuth_indx = 7
        track_index = 2
        file_labels[:,0] = file_labels[:,0]*efactor
    
    type_mask = CutMask(file_labels)
    vertex_mask = VertexMask(file_labels,azimuth_index=azimuth_index,track_index=track_index,max_track=track_max)
    vertex_cut = np.logical_and(vertex_mask[start_cut], vertex_mask[end_cut])
    mask = np.logical_and(type_mask[cut_name], vertex_cut)
    mask = np.array(mask,dtype=bool)

    # Make cuts for event type and zenith
    zenith = np.array(file_labels[:,1])
    energy =  np.array(file_labels[:,0])
    #oscweights = np.array(file_weights[:,-2])

    total_events = len(zenith)
    zenith=zenith[mask]
    energy = energy[mask]
    #oscweights=oscweights[mask]
    events_after_type_cut = len(zenith)
    energy_mask = np.logical_and(energy >= emin, energy <= emax)
    
    #if sum(energy_mask) == 0:
    #    print(energy[:10])
    zenith = zenith[energy_mask]
    energy = energy[energy_mask]
    #oscweights=oscweights[energy_mask]
    all_energy=np.concatenate((all_energy,energy))
    #all_weights=np.concatenate((all_weights,oscweights))
    all_zenith=np.concatenate((all_zenith,zenith))

    events_after_zenith_cut = len(zenith)
    print("%i\t %i\t %i\t"%(total_events,events_after_type_cut,events_after_zenith_cut))

    #Sort into bins and count
    zmin_array = np.ones((events_after_zenith_cut))*zmin
    zenith_bins = np.floor((zenith-zmin_array)/float(bin_size))
    count_zenith += np.bincount(zenith_bins.astype(int),minlength=bins)
min_number_events = min(count_zenith[0:-2])
min_bin = np.where(count_zenith==min_number_events)
if type(min_bin) is tuple:
    min_bin = min_bin[0][0]

"""
print("Minimum bin value %i events at %i GeV"%(min_number_events, zenith_bin_array[min_bin]))
print("Cutting there gives total events: %i"%(min_number_events*bins))
print(count_zenith)

afile = open("%s/final_distribution_zmin%.0fzmax%.0f_%s.txt"%(outdir,zmin,zmax,cut_name),"w")
afile.write("Minimum bin value %i events at %i GeV"%(min_number_events,zenith_bin_array[min_bin]))
afile.write('\n')
afile.write("Cutting there gives total events: %i"%(min_number_events*bins))
afile.write('\n')
afile.write("Bin\t Energy\t Number Events\n")
for index in range(0,len(count_zenith)):
    afile.write(str(index) + '\t' + str(int(zenith_bin_array[index])) + '\t' + str(int(count_zenith[index])) + '\n')
afile.close()
"""

"""
plt.figure(figsize=(10,8))
plt.title("Events Binned by %.3f"%bin_size)
plt.bar(zenith_bin_array,count_zenith,alpha=0.5,width=0.0314,align='edge')
plt.xlabel("zenith")
plt.ylabel("number of events")
plt.savefig("%s/ZenithDistribution_zmin%.0fzmax%.0f_%s.png"%(outdir,zmin,zmax,cut_name))
"""
#all_weights=all_weights/25712
print(np.max(all_energy))
plt.figure()
ecnts,ebins,eind=plt.hist(all_energy,bins=500,histtype='step') #,weights=all_weights)
plt.savefig("%s/EnergyDistribution.png"%(outdir))
"""
plt.clf()
plt.hist(all_zenith,bins=100) #,weights=all_weights)
plt.savefig("%s/WeightedZenithDistribution.png"%(outdir))
plt.clf()
all_zenith=np.cos(all_zenith)
plt.hist(all_zenith,bins=100)#,weights=all_weights)
plt.savefig("%s/CosZenithDistribution.png"%(outdir))
"""
afile = open("%s/final_distribution_emin%.0femax%.0f_%s.txt"%(outdir,emin,emax,cut_name),"w")
afile.write("Cutting there gives total events: %i"%(min_number_events*bins))
afile.write('\n')
afile.write("Bin\t Energy\t Number Events\n")
#print(ecnts)
#print(ebins)
for index,abin,cnt in zip(eind, ebins, ecnts):
    afile.write(str(index) + '\t' + str(int(abin)) + '\t' + str(int(cnt)) + '\n')
afile.close()

