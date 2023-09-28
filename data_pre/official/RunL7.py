#!/usr/bin/python
##Test oscNext_L6_atm_muon_variables STV and TH##
 
from __future__ import division
import math
import os

import icecube
from I3Tray import *
from operator import itemgetter

from icecube import lilliput
import icecube.lilliput.segments
import oscNext_L7_pid

from icecube import icetray, phys_services, dataclasses, dataio, photonics_service, TrackHits, StartingTrackVetoLE

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input_file",type=str,
                    dest="input_file",help="name of inpute file")
parser.add_argument("-o","--output_name",type=str,default=None,
                    dest="output_name",help="output name of file")
parser.add_argument("-id","--indir",default=None,
                    dest="indirectory",help="directory for folder in scratch")
parser.add_argument("-od","--outdir",default=None,
                    dest="outdirectory",help="directory for folder in scratch")
args = parser.parse_args()

infilename=args.indirectory + args.input_file
outfilename=args.outdirectory + args.output_name

file_list = []
if os.path.isfile(infilename):
    print("Working on %s"%infilename)
else:
    print("Cannot find %s"%infilename)
gcd_file = "/mnt/research/IceCube/gcd_file/GeoCalibDetectorStatus_AVG_55697-57531_PASS2_SPE_withScaledNoise.i3.gz"
file_list.append(gcd_file)
# for filename in glob.glob(data_file):
file_list.append(infilename)
    

#@icetray.traysegment
tray = I3Tray()
tray.AddModule("I3Reader","reader", FilenameList = file_list)
tray.AddSegment(oscNext_L7_pid.oscNext_L7,"L7pid",
                uncleaned_pulses="SplitInIcePulses",
                cleaned_pulses="SRTTWOfflinePulsesDC")
tray.AddModule('I3Writer', 'writer', Filename= outfilename+'.i3.zst')
tray.AddModule('TrashCan','thecan')
tray.Execute()
tray.Finish()

