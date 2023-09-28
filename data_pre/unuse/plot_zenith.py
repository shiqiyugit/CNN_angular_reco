import matplotlib
matplotlib.use('AGG')
import numpy as np
import sys
import os
import argparse
import glob
from icecube import icetray, dataio, dataclasses
from I3Tray import I3Units
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-r1", "--ratio1", type=int, default=1,
                    dest="ratio1")
parser.add_argument("-r2", "--ratio2", type=int, default=2,
                    dest="ratio2")
parser.add_argument("-r", "--ratio_plot", type=bool, default=False,
                    dest="ratio_plot")
parser.add_argument("-i", "--input_files",default='l3.files',
                    type=str,dest="input_files", help="names for input files")
parser.add_argument("-d", "--path",type=str,default='l3/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--outdir",type=str,default='./',
                    dest="outdir", help="out directory for plots")
parser.add_argument("-n", "--name",type=str,default="None",
                    dest="name", help="name for output folder")
parser.add_argument("-b", "--bin_size",type=float,default=0.314,
                    dest="bin_size", help="Size of energy bins in GeV (default = 1GeV)")
parser.add_argument("--emax",type=float,default=3.14,
                    dest="emax",help="Cut anything greater than this energy (in GeV)")
parser.add_argument("--emin",type=float,default=0.0,
                    dest="emin",help="Cut anything less than this energy (in GeV)")

args = parser.parse_args()
input_files = args.input_files
f=open(input_files,'r')

files_with_paths = f.readlines()
files_with_paths = map(lambda s: s.strip(), files_with_paths)
event_file_names = files_with_paths
emax = args.emax
emin = args.emin
bin_size = args.bin_size
energy_bins = int((emax-emin)/float(bin_size))
count_events = 0
energy = []
zenith = []
for event_file_name in event_file_names:
    event_file = dataio.I3File(event_file_name)
    print("reading file: {}".format(event_file_name))
    try:
      for frame in event_file:
       if frame.Stop == icetray.I3Frame.Physics:
#            print("im daq!")
            #header = frame["I3EventHeader"]
            #if frame["I3EventHeader"].sub_event_stream != "InIceSplit":
            #    continue
            nu_energy = frame["I3MCTree"][0].energy
            nu_zenith = frame["I3MCTree"][0].dir.zenith
            zenith.append(nu_zenith)
            energy.append(nu_energy)
            count_events +=1
#            print(zenith, energy)
    except Exception:
      print("Exception accur!")
      pass
zenith=np.array(zenith)
energy=np.array(energy)

print("zenith:",zenith.shape)
print("energy >80:", energy[energy>80].shape)

plt.figure()
plt.hist(energy, bins=100)

plt.show()
plt.savefig("test.png")



