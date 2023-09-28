#################################
# Plots input and output features for ONE file
#   Inputs:
#       -i input file:  name of ONE file
#       -d  path:       path to input files
#       -o  outdir:     path to output_plots directory or where final dir will be created
#       -n  name:       Name of directory to create in outdir (associated to filenames)
#       --emax:         Energy max cut, plot events below value, used for UN-TRANSFORM
#       --emin:         Energy min cut, plot events above value
#       --tfactor:         Track factor to multiply, use for UN-TRANSFORM
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
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

font = {
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
import argparse
import matplotlib.colors as colors
from matplotlib.pyplot import contour, contourf

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_file",default='prediction_values.hdf5',
                    type=str,dest="input_file", help="name for ONE input file")
parser.add_argument("-d", "--path",type=str,default=
#'/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/flat_zenith_all_start_all_end_flat_101bins_5_300_allics_mobilev2_linear/oscnext_340epochs/',
#'/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/flat_zenith_all_start_end_IC19_flat_101bins_26500evtperbin_5_300_sigmoid/oscnext_512epochs/',
#'/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/flat_zenith_contained_flat_101bins_22900evtperbin_5_300_largeweighted_v2/oscnext_360epochs/',
#'/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/flat_zenith_contained_flat_101bins_22900evtperbin_5_300_largeweighted/oscnext_736epochs/',
#'/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/flat_zenith_all_start_all_end_flat_101bins_21500evtperbin_5_300_weighted/oscnext_600epochs/',
#'/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/flat_zenith_all_start_all_end_flat_101bins_5_300_linear_oscweight_mobilev2/oscnext_1000epochs/',
#'/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/flat_zenith_IC_start_all_end_flat_101bins_5_300_linear_fresh/oscnext_500epochs/',
#'/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/flat_zenith_IC_start_IC_end_flat_101bins_5_300_ABC_linear/oscnext_680epochs/',
'/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/energy_contained_start_contained_end_flat_101bins_1_500_allics_3Ddropout_0002/retroL6_1110epoch/',
#'/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/flat_zenith_all_start_all_end_flat_101bins_5_300_linear_ABC/oscnext_660epochs/',
#flat_zenith_all_start_all_end_flat_101bins_5_300_linear_ABC_weighted/oscnext_500epochs/',

#'/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/flat_zenith_IC_start_all_end_flat_101bins_5_300_wei_fresh/oscnext_610epochs/',

#'/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/lrEpochs64_lrInit0.001_lrDrop0.6_weighted/oscnext_656epochs/',

                    dest="path", help="path to input files")
#parser.add_argument("-o", "--outdir",type=str,default='/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots/numu_flat_Z_5_150_CC_vertexDC_lrEpochs40_containedIC19_tanh_v0/oscnext_136epochs/',
#                    dest="outdir", help="out directory for plots")
parser.add_argument("-n", "--name",default='in_out',
                    dest="name", help="name for output folder")
parser.add_argument("--filenum",default=1,
                    dest="filenum", help="number for file, if multiple with same name")
parser.add_argument("--emax",type=float,default=300,
                    dest="emax",help="Cut anything greater than this energy (in GeV)")
parser.add_argument("--emin",type=float,default=1,
                    dest="emin",help="Cut anything smaller than this energy (in GeV)")

parser.add_argument("--tfactor",type=float,default=200.0,
                    dest="tfactor",help="Multiplication factor for track length")
parser.add_argument("-c", "--cuts",type=str, default="CC",
                    dest="cuts", help="Type of events to keep (all, cascade, track, CC, NC, etc.)")
parser.add_argument("--do_cuts", default=False,action='store_true',
                        dest='do_cuts',help="Apply cuts! Don't ignore energy and event cut")
parser.add_argument("--trackcut", default=False, action='store_true',
                        dest='trackcut', help='apply cut on track length')
parser.add_argument("--vcut", default=None, type=str, 
                        dest='vertex_cut', help="apply vertex true cut: all, IC")
parser.add_argument("--zmin",type=float,default=0,#2.15,
                    dest="zmin",help="Cut anything less than this (zenith)")
parser.add_argument("--zmax",type=float,default=3.15,#3.15,
                    dest="zmax",help="Cut anything greater than this (zenith)")
parser.add_argument("--cut_reco", default=False,
                        dest='cut_reco',help="cut on pred cnn z value")
parser.add_argument("--cut_true", default=False,
                        dest='cut_true',help="cut on true z value")
parser.add_argument("--cut_res", default=False,
                        dest='cut_res',help="cut on resolution")
parser.add_argument("--zenith_res", default=0.5,
                        dest='zenith_res',help="zenith resolution cut, absolute difference")
parser.add_argument("--input_merged", default=False,dest='input_merged',help="merged input")

args = parser.parse_args()

do_output = True
do_input = True
file_was_transformed = True
merged=args.input_merged

input_file = args.path + args.input_file
name = args.name
outdir = args.path + name
if os.path.isdir(outdir) != True:
    os.mkdir(outdir)
print("Saving plots to %s"%outdir)

if args.filenum:
    filenum = str(args.filenum)
else:
    filenum=args.filenum
tfactor = args.tfactor

zenith_min = args.zmin
zenith_max = args.zmax
zenith_res = args.zenith_res

energy_max = args.emax
energy_min = args.emin
cut_name = args.cuts
do_cuts = args.do_cuts
file_name="nocut"
if do_cuts: 
  file_name="_z_%.2f_%.2f_E_%3.2f_%3.2f"%(zenith_min,zenith_max,energy_min, energy_max)


# Here in case you want only one (input can take a few min)
strings_19 = [17, 18, 19, 25, 26, 27, 28, 34, 35, 36, 37, 38, 44, 45, 46, 47, 54, 55, 56]
strings_8 = [84, 85, 79, 80, 83, 86, 81, 82]
colors_list=plt.get_cmap('tab20')

f = h5py.File(input_file, 'r')
if file_was_transformed:
    Y_test = f['Y_test'][:]
    Y_pred = f['Y_predicted'][:]
#    weight_test = f['weight_test'][:]

    X_test_DC = f['X_test_DC'][:]
    X_test_IC = f['X_test_IC1'][:]
    if merged: 
      X_test = np.concatenate((X_test_DC,X_test_IC),axis=1)
    print("test size: ", Y_test.shape)
else:
    Y_test = f['labels'][:]
    X_test_DC = f['features_DC'][:]
    X_test_IC = f['features_IC'][:]
try: 
  weight_test = f['weights_test'][:]
  weight_train = f['weights_train'][:]
  weight_validate =f['weights_validate'][:]

except:
  weight_test = None
  weight_train = None
  weight_validate = None

try:
    Y_train = f['Y_train'][:]
    X_train_DC = f['X_train_DC'][:]
    X_train_IC = f['X_train_IC'][:]

    Y_validate = f['Y_validate'][:]
    X_validate_DC = f['X_validate_DC'][:]
    X_validate_IC = f['X_validate_IC'][:]
    print("Train shape: ", Y_train.shape)
except:    
    Y_train = None
    X_train_DC = None
    X_train_IC = None

    Y_validate = None
    X_validate_DC = None
    X_validate_IC = None

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

if Y_train is None: #Test only file
    Y_labels = Y_test
    Y_labels_pred = Y_pred
    weights = weight_test
    if merged:
      X = X_test
    else:
      X_DC = X_test_DC
      X_IC = X_test_IC
else:
    Y_labels = np.concatenate((Y_test,Y_train,Y_validate))
    X_DC = np.concatenate((X_test_DC,X_train_DC,X_validate_DC))
    X_IC = np.concatenate((X_test_IC,X_train_IC,X_validate_IC))
    weights=np.concatenate((weight_test, weight_train, weight_validate))
efactor=100
# Untransform so energy and track are in original range. NOTE: zenith still cos(zenith)
if file_was_transformed:
    print("MULTIPLYING ENERGY BY %f and TRACK BY %f to undo transform"%(efactor,tfactor))
    print("ASSUMING TRACK IS AT INDEX 2")
    Y_labels[:,0] = Y_labels[:,0]*efactor
    Y_labels[:,2] = Y_labels[:,2]*tfactor

if reco_test is not None:
    reco_labels = np.concatenate((reco_test,reco_train,reco_validate))

#zenith=Y_labels[:,1]
#uvalues, index, unique, counts=np.unique(zenith, return_inverse=True, return_index=True, return_counts=True)

#true=0.021496966
#print("true is 0.021496966")
#index=np.where(zenith==true)
#print("labels: ", Y_labels[index[0][0]])
#print("labels: ", Y_labels[index[0][1]])
from utils.PlottingFunctions import plot_input_3D
#plot_input_3D(X_IC, X_DC,Y_labels,Y_labels_pred,"./",evtInd=index[0][0],varInd=0,filename="3D1")
#plot_input_3D(X_DC, X_IC,Y_labels,Y_labels_pred,"./",evtInd=index[0][1],varInd=0,filename="3D2")

# Apply Cuts
from handle_data import CutMask
from utils.handle_data import VertexMask

if do_cuts:

    print("CUTTING ON Energy [%f,%f] AND EVENT TYPE %s"%(energy_min, energy_max, cut_name))
    mask = CutMask(Y_labels)
    cut_energy = np.logical_and(Y_labels[:,0] >= energy_min, Y_labels[:,0] <= energy_max)

    if args.trackcut:
      cut_track=Y_labels[:,2]>=100
      cut_energy=np.logical_and(cut_energy,cut_track)
    cut=cut_energy
    if args.cut_true:
      print("CUTTING ON True Zenith [%f,%f] "%(zenith_min,zenith_max))
      cut_zenith = np.logical_and(Y_labels[:,1] >= zenith_min, Y_labels[:,1]<=zenith_max)
      cut = np.logical_and(cut_zenith, cut_energy)

    if args.cut_reco:
      print("CUTTING ON CNN Zenith [%f,%f] "%(zenith_min,zenith_max))
      cut_zenith = np.logical_and(Y_labels_pred[:,0] >= zenith_min, Y_labels_pred[:,0]<=zenith_max)
      cut = np.logical_and(cut_zenith, cut)

    if args.cut_res:
      print("Cutting on absolute difference of Zenith: %f"%zenith_res)
      cut_zenith = np.abs(Y_labels_pred[:,0]-Y_labels[:,1])>=zenith_res
      cut = np.logical_and(cut_zenith, cut)
    
    mask_cuts=mask[cut_name]
    if args.vertex_cut is not None:
      print("Cutting on vtx: ", args.vertex_cut)
      vtxmask=VertexMask(Y_labels,azimuth_index=7,track_index=2,max_track=1.)
      print("before vtx cut: ",Y_labels.shape[0])
      print("after vtx cut: ", Y_labels[vtxmask["contained_IC19_start"]].shape[0])
      if args.vertex_cut == "contained":
        allvtxmask=np.logical_and(vtxmask["contained_IC19_end"], vtxmask["contained_IC19_start"])
      elif args.vertex_cut == "IC19":
        allvtxmask=np.logical_and(vtxmask["end_IC19"], vtxmask["start_IC19"])
      elif args.vertex_cut == "noendIC19":
        allvtxmask=np.logical_not(vtxmask["contained_IC19_end"])
      else:
        allvtxmask = np.logical_and(vtxmask["all_end"], vtxmask["all_start"])

      mask_cuts=np.logical_and(mask[cut_name],allvtxmask)

    all_cuts = np.logical_and(mask_cuts, cut)
    Y_labels_cut = Y_labels[all_cuts]
    Y_labels_pred_cut = Y_labels_pred[all_cuts]
    if weights is not None:
      weights=weights[all_cuts]
#    for ind in range(50):
#      run=weights[ind][0]
#      subrun=weights[ind][1]
#      evt=weights[ind][2]
#      print(run, subrun, evt)
    if merged:
      X_cut = X[all_cuts]
    else:
      X_DC_cut = X_DC[all_cuts]
      X_IC_cut = X_IC[all_cuts]
    if reco_test is not None:
        reco_labels = reco_labels[all_cuts]
    print(Y_labels.shape, Y_labels_cut.shape, Y_labels_pred.shape, Y_labels_pred_cut.shape)

def plot_output(Y_values, Y_values_cut, outdir,filenumber=None,transformed=file_was_transformed, filename=""):
    if file_was_transformed:
        names = ["Energy", "Zenith", "Track Length", "Time", "X", "Y", "Z", "Azimuth"]
        units = ["(GeV)", "(rad)", "(m)", "(s)", "(m)", "(m)", "(m)", "(rad)"]
    else:
        names = ["Energy", "Zenith", "Azimuth", "Time", "X", "Y", "Z", "Track Length"]
        units = ["(GeV)", "(rad)", "(rad)", "(s)", "(m)", "(m)", "(m)", "(m)"]
    plt.figure(figsize=(10,7))
    for i in range(0,len(names)):
        plt.clf()
        '''
        plt.subplots_adjust(hspace=0.3)
        plt.subplot(311)
        nocut,nocutbins,img=plt.hist(Y_values[:,i],bins=100, log=False);
        plt.title("%s Distribution"%names[i],fontsize=25)
        plt.ylabel("No Cut")
        plt.yticks(fontsize=15)
        plt.subplot(312)
        cuts,cbins,cimg=plt.hist(Y_values_cut[:,i],bins=nocutbins, log=False);
        plt.ylabel("Cut")

        plt.subplot(313)
        ratio=cuts/nocut
        plt.plot(nocutbins[0:-1],ratio,'+')
#bins=nocutbins, log=False);
        plt.xlabel("%s %s"%(names[i],units[i]),fontsize=15)
        plt.xticks(fontsize=15)
        plt.ylabel("Cut/NoCut")
        plt.yticks(fontsize=15)
        '''
        con,bins,figs=plt.hist(Y_values[:,i],bins=100);
        plt.ylim(0, 1.3*np.max(con))
        plt.ylabel("Number of events")
        plt.xlabel("%s %s"%(names[i],units[i]))
        textstr = 'IceCube Work in Progress'
        ax=plt.gca()
        plt.text(0.48, .950, textstr, transform=ax.transAxes,color='gray')
        if filenum:
            filenum_name = "_%s"%filenum
        else:
            filenum_name = ""
        plt.savefig("%s/Output_%s%s%s.png"%(outdir,names[i].replace(" ", ""),filenum_name,filename), bbox_inches='tight')

    num_events = Y_values.shape[0]
    flavor = list(Y_values[:,9])
    print("Fraction NuMu: %f"%(flavor.count(14)/num_events))
    print("Fraction Track: %f"%(sum(Y_values[:,8])/num_events))
    print("Fraction Antineutrino: %f"%(sum(Y_values[:,10])/num_events))
    print("Fraction CC: %f"%(sum(Y_values[:,11])/num_events))

def get_stats(xaxis, true, pred, bins=[20,20]):
    diff=(pred-true)
    x=xaxis
    plt.figure()
    nomi,xbin,ybin,img=plt.hist2d(x, diff, bins=bins,weights=diff)
    denom,x_,y_,img_=plt.hist2d(x, diff, bins=bins)
    denom[denom==0.0]=1
    nomi[denom==0]=0
    avgs=nomi/denom

    mean=np.average(avgs,axis=1)
    std=[mean-np.quantile(avgs,0.16,axis=1),np.quantile(avgs,0.84,axis=1)-mean]
#    std=np.std(avgs,axis=1)
#    print(mean,std)
    return mean, std, xbin, ybin

def plot_output_zenith(labels, preds, outdir,filenumber=None,transformed=file_was_transformed,filename="", weights=None):

    names = ["Zenith (rad)"]
    units = [""]
    bins=[i*0.0315 for i in range(100)]

    if filename == "CosZenith":
      names = ["cos(zenith)"]
      labels[:,1]=np.cos(labels[:,1])
      preds[:,0]=np.cos(preds[:,0])
      bins=[i*0.01 for i in range(100)]

    plt.figure(figsize=(10,7))
    tconts,tbin,tbar=plt.hist(labels[:,1],bins=bins, color='g',label="True",alpha=0.5, weights=weights)#, density=True);
    pconts,pbin,pbar=plt.hist(preds[:,0],bins=bins, color='b',label="Pred",alpha=0.5, weights=weights)#, density=True);
    ymax=max(np.max(tconts), np.max(pconts))
    plt.ylim(0,ymax*1.2)
    plt.xlabel("%s %s"%(names[0],units[0]),fontsize=22)
    plt.ylabel("Number of events")
    textstr = 'IceCube Work in Progress'
    ax=plt.gca()
    plt.text(0.48, .950, textstr, transform=ax.transAxes,color='gray')
    plt.legend(loc='center')
    if filenum:
        filenum_name = "_%s"%filenum
    else:
        filenum_name = ""
    plt.savefig("%s/Output_true_vs_pred_zenith_%s%s%s.png"%(outdir,names[0].replace(" ", ""),filenum_name,filename))
    plt.close
    
    print("calculating weights...")
    ratioweights=tconts/pconts
    afile = open("%s/weights.txt"%(outdir),'w')
    for b, c in zip(bins, ratioweights):
        afile.write(str(b)+'\t')
        afile.write(str(c) + '\t')
        afile.write('\n')
    afile.close()
    

    resolution=preds[:,0]-labels[:,1]
    fig, ax=plt.subplots(figsize=(10,7)) #plt.figure()
    ax.hist(resolution,bins=50)
    tot=resolution.shape[0]
    meanres=np.mean(resolution)
    rmsres=np.sqrt(np.mean(resolution ** 2))
    textstr = '\n'.join((
            r'$\mathrm{evt}=%d$'%(tot,),
            r'$\mathrm{mean}=%.2f$' % (meanres, ),
            r'$\mathrm{RMS}=%.2f$' % (rmsres, )))
    props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
    ax.text(0.8, 0.95, textstr, transform=ax.transAxes, fontsize=20,
        verticalalignment='top', bbox=props)

    plt.xlabel("CNN - True Zenith",fontsize=20)
    plt.ylabel("Event", fontsize=20)
    plt.savefig("%s/Output_resolution_%s.png"%(outdir,filename))
    plt.close

    plt.figure()
#    nomi,xbin,ybin,img=plt.hist2d(labels[:,1], resolution,bins=[30,20],weights=resolution)
#    denom,x_,y_,img_=plt.hist2d(labels[:,1], resolution,bins=[30,20])
#    denom[denom==0.0]=1
#    avgs=nomi/denom
#    mean=np.average(avgs,axis=1)
    mean,std,xbin,ybin=get_stats(labels[:,1],labels[:,1], preds[:,0])
    plt.clf()
    plt.errorbar(xbin[0:-1],mean, yerr=std,fmt='b+')
    plt.xlabel("True Zenith", fontsize=20)
    plt.ylabel("CNN - True Zenith", fontsize=20)
    plt.savefig("%s/Output_bias_vs_true_Zenith%s.png"%(outdir,filename))

    mean,std,xbin,ybin=get_stats(labels[:,6],labels[:,1], preds[:,0],[50,20])
    plt.clf()
    plt.errorbar(xbin[0:-1],mean, yerr=std,fmt='+')
    plt.xlabel("True Z", fontsize=20)
    plt.ylabel("CNN - True Zenith", fontsize=20)
    plt.savefig("%s/Output_bias_vs_true_Z%s.png"%(outdir,filename))

    mean,std,xbin,ybin=get_stats(labels[:,2], labels[:,1], preds[:,0],[50,50])
    plt.clf()
    plt.errorbar(xbin[0:-1],mean, yerr=std,fmt='+')
    plt.xlabel("True TrackLength (m)", fontsize=20)
    plt.ylabel("CNN - True Zenith", fontsize=20)
    plt.savefig("%s/Output_bias_vs_true_TrackLength%s.png"%(outdir,filename))

    plt.clf()
    plt.hist2d(labels[:,6],resolution,bins=[xbin,ybin])#, range=[[-1000,0],[-1,1]])
    plt.errorbar(xbin[0:-1],mean, yerr=std,fmt='+')
    plt.xlabel("True Z")
    plt.ylabel("CNN -True Zenith")
    plt.savefig("%s/Output_bias_vs_true_Z2D%s.png"%(outdir,filename))

    plt.clf()
    plt.hist2d(labels[:,6],preds[:,0], bins=[100,100])
    plt.xlabel("True Z")
    plt.ylabel("CNN Zenith")
    plt.savefig("%s/Output_CNNZenith_vs_true_Z%s.png"%(outdir,filename))

    plt.clf()
    plt.hist2d(labels[:,6],labels[:,1],bins=[100,100])
    plt.xlabel("True Z")
    plt.ylabel("True Zenith")
    plt.savefig("%s/Output_TrueZenith_vs_true_Z%s.png"%(outdir,filename))

    
#    plt.figure()
#    plt.hist2d(labels[:,1],preds[:,0],bins=[bins,bins])
#    plt.ylabel("CNN Zenith",fontsize=10)
#    plt.xlabel("True Zenith",fontsize=10)
#    plt.savefig("%s/Output_true_vs_pred_zenith_2D_%s%s%s.png"%(outdir,names[0].replace(" ", ""),filenum_name,filename))

def plot_energy_zenith(Y_values, Y_values_pred, outdir,filenumber=None,filename=""):
    
    plt.figure()
    zcts,zx,zy,img=plt.hist2d(Y_values[:,1], Y_values[:,6], bins=[50,50],cmap="Blues")
    plt.title(filename)
    plt.xlabel("True Zenith")
    plt.ylabel("True Z (m)")
    cbar = plt.colorbar()
    plt.savefig("%s/Z_Zenith%s.png"%(outdir,filename))
    plt.clf()

    plt.hist2d(Y_values_pred[:,0], Y_values[:,6], bins=[50,50],cmap="Reds")
    plt.title(filename)
    plt.xlabel("CNN Zenith")
    plt.ylabel("True Z (m)")    
    cbar = plt.colorbar()
    plt.savefig("%s/Z_CNNZenith%s.png"%(outdir,filename))

    plt.clf()
    plt.hist2d(Y_values[:,2], Y_values[:,6], bins=[50,50],cmap="Greens")
    plt.title(filename)
    plt.xlabel("True Track (m)")
    plt.ylabel("True Z (m)")
    cbar = plt.colorbar()
    plt.savefig("%s/Track_Z%s.png"%(outdir,filename))

    plt.clf()
    plt.hist2d(Y_values_pred[:,0], Y_values[:,2], bins=[50,50],cmap="Oranges")
    plt.title(filename)
    plt.ylabel("True Track (m)")
    plt.xlabel("CNN Zenith")
    cbar = plt.colorbar()
    plt.savefig("%s/Track_CNNZenith%s.png"%(outdir,filename))

    plt.clf()
    plt.hist2d(Y_values[:,1],Y_values[:,2], bins=[50,50],cmap="Purples")
    plt.title(filename)
    plt.ylabel("True Track (m)")
    plt.xlabel("True Zenith")
    cbar = plt.colorbar()
    plt.savefig("%s/Track_TrueZenith%s.png"%(outdir,filename))

    '''
    plt.figure()
    ymeans=[]
    ystds=[]
    ymedian=[]
    import itertools
    for i in range(zcts.shape[0]):
      print(zy[0:-1].shape, zcts[i].shape)
      ymean=np.average(zy[0:-1],weights=zcts[i])
      ymeans.append(ymean)
      expanded=list(list(itertools.repeat(zy[j],int(zcts[i][j]))) for j in range(zcts[i].shape[0]))
      medianlist = [x for t in expanded for x in t]
      npa = np.array(medianlist)
      median = np.std(npa)
      ymedian.append(median)
    ystds=ymedian
    plt.errorbar(zx[0:-1], ymeans, yerr=ystds, fmt='o', color='black',
             ecolor='lightgray', elinewidth=3, capsize=0);
    plt.savefig("%s/Z_Zenith_Profile%s.png"%(outdir,filename))
    '''

    plt.figure()
    cts,xbin,ybin,img = plt.hist2d(Y_values[:,0], Y_values[:,1], bins=[25,100],cmap='bwr')
    bins=[xbin,ybin]
    plt.ylabel("True Neutrino Zenith")
    plt.xlabel("True Neutrino Energy (GeV)")
    plt.savefig("%s/Energy_trueZenith%s.png"%(outdir,filenum))
  
    
    plt.figure()
    counts,xbinp,ybinp,imgp = plt.hist2d(Y_values[:,0], Y_values_pred[:,0], bins=bins)

    plt.subplots_adjust(wspace=0.3)

    plt.subplot(121)
    cs=contourf(cts.T, extent=[xbin.min(),xbin.max(),ybin.min(),ybin.max()],linewidths=2,cmap='bwr')#,norm=norm)
    plt.ylabel("True Neutrino Zenith",fontsize=20)
    plt.xlabel("True Neutrino Energy (GeV)")
    cbar = plt.colorbar()

    plt.subplot(122)
    cs2=contourf(counts.T,extent=[xbinp.min(),xbinp.max(),ybinp.min(),ybinp.max()],linewidths=2,cmap='bwr')#,norm=norm)
    plt.ylabel("CNN Neutrino Zenith",fontsize=20)
    plt.xlabel("True Neutrino Energy (GeV)")
    cbar = plt.colorbar()
#    cbar.set_clim(0, zmax)

    plt.savefig("%s/Energy_Zenith%s.png"%(outdir,filename))
    plt.close
    

def plot_input_zenith(Y_values,X_values_DC,X_values_IC,outdir,filenumber=None, filename=""):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import pandas as pd
    import seaborn as sns

#    name = ["Charge (p.e.)", "Raw Time of First Pulse (ns)", "Raw Time of Last Pulse (ns)", "Charge weighted mean of pulse times", "Charge weighted std of pulse times"]
    DC_data=X_values_DC[:].flatten()
    X = np.reshape(DC_data, (48066,2400))
#    DC_data = np.reshape(X_values_DC[:,0], (48066,-1))
    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(X,columns=feat_cols)
    Y_values=np.floor(Y_values/0.314)
    df['y']=Y_values
    
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)

    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]
    rndperm = np.random.permutation(df.shape[0])

    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(\
    xs=df.loc[rndperm,:]["pca-one"], \
    ys=df.loc[rndperm,:]["pca-two"], \
    zs=df.loc[rndperm,:]["pca-three"], \
    c=df.loc[rndperm,:]["y"], \
    cmap='tab10')
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()

    plt.savefig("%s/3D_charge.png"%(outdir))
    sns.scatterplot(x="pca-one",y="pca-two",\
    hue="y",\
    palette=sns.color_palette("hls", 10),\
    data=df.loc[0:500,:],legend="full",alpha=0.3)
    plt.savefig("%s/PCA_charge.png"%(outdir))

    df_subset=df.loc[0:1000,:].copy()
    data_subset = df_subset[feat_cols].values

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(data_subset)
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(\
    x="tsne-2d-one", y="tsne-2d-two",\
    hue="y",\
    palette=sns.color_palette("hls", 10),\
    data=df_subset,\
    legend="full",\
    alpha=0.3\
    )
    plt.savefig("%s/TSNE_2D_charge.png"%(outdir))

def plot_input(X_values_DC,X_values_IC,outdir,filenumber=None, filename=""):
    name = ["Charge (p.e.)", "Raw Time of First Pulse (ns)", "Raw Time of Last Pulse (ns)", "Charge weighted mean of pulse times", "Charge weighted std of pulse times"]
    IC_label = "IC"
    DC_label = "DC"
    
    print(X_values_DC.shape,X_values_IC.shape)
    for i in range(0,X_values_DC.shape[-1]):
       
        DC_data = X_values_DC[...,i].flatten()
        IC_data = X_values_IC[...,i].flatten()

        min_range = min(min(DC_data),min(IC_data))
        max_range = max(max(DC_data),max(IC_data))
        plt.figure()
        plt.hist(IC_data,log=True,bins=100,range=[min_range,max_range],color='g',label=IC_label,alpha=0.5);
        plt.hist(DC_data,log=True,bins=100,range=[min_range,max_range],color='b',label=DC_label,alpha=0.5);
        plt.title(name[i],fontsize=25)
        plt.xlabel(name[i],fontsize=15)
        plt.legend(fontsize=15)
        if filenum:
            filenum_name = "_%s"%filenum
        else:
            filenum_name = ""
        plt.savefig("%s/Input_Variable%i%s%s.png"%(outdir,i,filenum_name,filename))
        plt.close

def plot_charges_vs_energy(X_values_DC,X_values_IC,Y_labels, outdir,filenumber=None, filename=""):

    DC_data = X_values_DC[...,0]
    IC_data = X_values_IC[...,0]

    DC_data=np.reshape(DC_data, (DC_data.shape[0],-1))
    IC_data=np.reshape(IC_data, (IC_data.shape[0],-1))

    DC_charge=np.sum(DC_data,axis=1)
    IC_charge=np.sum(IC_data, axis=1)
    print(DC_charge.shape, DC_charge[0:10])
    charges=DC_charge+IC_charge
    energy=Y_labels[:,0]
    
    plt.figure()
    plt.hist(DC_charge, 295, stacked=True, label="DC")
    plt.hist(IC_charge, 295, stacked=True, label="IC")
    plt.hist(charges, 295, alpha=0.5, label="DC+IC",histtype='step')
    plt.legend()
#    plt.hist(energy, 295, alpha=0.5, label="True Energy", histtype='step')
    plt.savefig("%s/Input_Charges_vs_TrueE%s.png"%(outdir,filename))

#    ratio=energy/charges
#    plt.figure()
#    plt.hist(ratio,100)
#    plt.savefig("%s/Input_Charges_vs_TrueE_ratio%s.png"%(outdir,filename))

def plot_input_strings(X_values_IC,X_values_IC_cut,outdir,filenumber=None, filename="",cutrange=""):
    name = ["Charge (p.e.)", "Raw Time of First Pulse (ns)", "Raw Time of Last Pulse (ns)", "Charge weighted mean of pulse times", "Charge weighted std of pulse times"]
    if filename == "IC":
       strings = strings_19
    elif filename == "DC":
       strings = strings_8

    print(X_values_IC.shape,X_values_IC_cut.shape)
    for i in range(0,X_values_IC.shape[-1]):
     fig, axs = plt.subplots(2, 2,tight_layout=True)
     axs[0,0].set_title("nocut",fontsize=12)
     axs[0,1].set_title("cut: %s"%cutrange,fontsize=12)
     axs[1,0].set_title("cut/nocut ratio",fontsize=12)
     for j in range(0,X_values_IC.shape[1]):
        IC_data_cut = X_values_IC_cut[:,j,:,i].flatten()
        IC_data = X_values_IC[:,j,:,i].flatten()

        min_range = min(min(IC_data),min(IC_data_cut))
        max_range = max(max(IC_data), max(IC_data_cut))
        ns0, bins0, patches0 = axs[0,0].hist(IC_data,log=True,bins=100,range=[min_range,max_range],color=colors_list(j),label="%i"%strings[j], histtype='step', alpha=0.8);
        ns1, bins1, patches1 = axs[0,1].hist(IC_data_cut,log=True,bins=100,range=[min_range,max_range],color=colors_list(j),label="%i"%strings[j], histtype='step', alpha=0.8);
        axs[1,0].plot(bins0[0:-1],ns1/ns0,color=colors_list(j),label="%i"%strings[j], marker='o', alpha = 0.8)
        axs[1,0].set_ylim(-0.1,1.2)        
     h, l =axs[0,0].get_legend_handles_labels()
     axs[1,1].legend(h, l, loc='center', fontsize=10,ncol=3)
     axs[1,1].plot()
     plt.xlabel(name[i],fontsize=15)
     if filenum:
         filenum_name = "_%s"%filenum
     else:
         filenum_name = ""
     plt.savefig("%s/Input_Variable%i%s_%s_%s.png"%(outdir,i,filenum_name,filename,cutrange))
     plt.clf()
     plt.close
     del fig, axs

def plot_DOM_range(X_values,Y_labels,X_values_cut=None, Y_labels_cut=None, outdir="./",filenumber=None, filename="",cutname=""):
    name = ["Charge (p.e.)", "Raw Time of First Pulse (ns)", "Raw Time of Last Pulse (ns)", "Charge weighted mean of pulse times", "Charge weighted std of pulse times"]
    X_vals=X_values[:,:,:,0]
    if X_values_cut is not None:
      X_vals_cut = X_values_cut[:,:,:,0]
    else:
      X_vals_cut = X_vals  

    if filename == "IC":
       strings = strings_19
    else:
       strings = strings_8
    print(X_vals.shape,X_vals_cut.shape)

    true_z=Y_labels[:,6]

    first_dom=[]
    dom_range=[]
    true_zs=[]
    doms=[]
    doms_cut=[]
    for i in range(X_vals.shape[0]):
      string_ind,dom_ind=np.nonzero(X_vals[i,:,:])

      if dom_ind.shape[0]>0:
        doms.append(len(dom_ind))

        first_dom.append(np.min(dom_ind))
        dom_range.append(np.max(dom_ind)-np.min(dom_ind)+1)
        true_zs.append(true_z[i])
    for i in range(X_vals_cut.shape[0]):
      string_ind_cut,dom_ind_cut=np.nonzero(X_vals_cut[i,:,:])
      if dom_ind.shape[0]>0:
        doms_cut.append(len(dom_ind_cut))

    plt.figure()
    plt.subplots_adjust(hspace=0.3)

    plt.subplot(211)
    nocut,bins,img=plt.hist(doms,histtype='bar',alpha=0.5,label="nocut",bins=160)
    cut,bins,img=plt.hist(doms_cut,histtype='bar',bins=bins,alpha=0.5,label="cut")
    plt.legend(loc='upper right', fontsize=15)

    plt.subplot(212)
    ratio=cut/nocut
    plt.plot(bins[0:-1],ratio)
    
    plt.savefig("%s/%s_num_DOMs_per_evt%s.png"%(outdir,filename,cutname))
    
    
    plt.hist(first_dom,histtype='bar') 
    binarr=[i for i in range(0,61)]
    plt.savefig("%s/%s_nonempty_fisrt_DOMind%s.png"%(outdir,filename,cutname))
    plt.clf()
    print(np.max(dom_range))
    plt.hist(dom_range,bins=binarr,histtype='bar')
    plt.savefig("%s/%s_nonempty_DOM_range.png"%(outdir,filename))

    '''
    plt.clf()
    plt.hist2d(true_zs,first_dom,bins=[100,60])
    plt.savefig("%s/%s_nonempty_first_DOMind_vs_trueZ.png"%(outdir,filename))

    plt.clf()
    plt.hist2d(true_zs,dom_range,bins=[100,60])
    plt.savefig("%s/%s_nonempty_DOM_range_vs_trueZ.png"%(outdir,filename))
    '''
 
def plot_DC_IC_strings(X_values_IC,X_values_IC_cut,X_IC_36, X_IC_36_cut, outdir,filenumber=None, filename="",cutrange=""):
    name = ["Charge (p.e.)", "Raw Time of First Pulse (ns)", "Raw Time of Last Pulse (ns)", "Charge weighted mean of pulse times", "Charge weighted std of pulse times"]
    if filename == "IC":
       strings = strings_19
    elif filename == "DC":
       strings = strings_8

    print(X_values_IC.shape,X_values_IC_cut.shape)
    for i in range(0,X_values_IC.shape[-1]):
     fig, axs = plt.subplots(2, 2,tight_layout=True)
     axs[0,0].set_title("nocut",fontsize=12)
     axs[0,1].set_title("cut: %s"%cutrange,fontsize=12)
     axs[1,0].set_title("cut/nocut ratio",fontsize=12)
     IC_36 = X_IC_36[:,9,:,i].flatten()
     IC_36_cut = X_IC_36_cut[:,9,:,i].flatten()

     for j in range(0,X_values_IC.shape[1]):
        IC_data_cut = X_values_IC_cut[:,j,:,i].flatten()
        IC_data = X_values_IC[:,j,:,i].flatten()

        min_range = min(min(IC_data),min(IC_data_cut))
        max_range = max(max(IC_data), max(IC_data_cut))
        ns0, bins0, patches0 = axs[0,0].hist(IC_data,log=True,bins=100,range=[min_range,max_range],color=colors_list(j),label="%i"%strings[j], histtype='step', alpha=0.8);
        ns1, bins1, patches1 = axs[0,1].hist(IC_data_cut,log=True,bins=100,range=[min_range,max_range],color=colors_list(j),label="%i"%strings[j], histtype='step', alpha=0.8);
        axs[1,0].plot(bins0[0:-1],ns1/ns0,color=colors_list(j),label="%i"%strings[j], marker='o', alpha = 0.8)
        axs[1,0].set_ylim(-0.1,1.2)

     axs[0,0].hist(IC_36,log=True,bins=100,range=[min_range,max_range],color='k',label="IC 36", histtype='step', alpha=0.8);
     axs[0,1].hist(IC_36_cut,log=True,bins=100,range=[min_range,max_range],color='k',label="IC 36", histtype='step', alpha=0.8);

     h, l =axs[0,0].get_legend_handles_labels()
     axs[1,1].legend(h, l, loc='center', fontsize=10,ncol=3)
     axs[1,1].plot()
     plt.xlabel(name[i],fontsize=15)
     if filenum:
         filenum_name = "_%s"%filenum
     else:
         filenum_name = ""
     plt.savefig("%s/Input_DC_IC36_Variable%i%s_%s_%s.png"%(outdir,i,filenum_name,filename,cutrange))
     plt.clf()
     plt.close
     del fig, axs


if do_output:
    print("Plotting outputs...")
#    plot_input_3D(X_DC, X_IC,Y_labels,Y_labels_pred,outdir,evtInd=828,varInd=3,filename="3D")
#    plot_input_3D(X_DC, X_IC,Y_labels,Y_labels_pred,outdir,evtInd=22952,varInd=3,filename="3D")
#    plot_DOM_range(X_DC, Y_labels,outdir=outdir,filenumber=filenum,filename="DC",cutname="nocut")
#    plot_DOM_range(X_IC,Y_labels,outdir=outdir,filenumber=filenum,filename="IC",cutname="nocut")

    plot_output(Y_labels, Y_labels, outdir,filenumber=filenum,filename="nocut")
    plot_output_zenith(labels=Y_labels, preds=Y_labels_pred, outdir=outdir,filenumber=filenum,filename="Zenith")
#    plot_output_zenith(labels=Y_labels, preds=Y_labels_pred, outdir=outdir,filenumber=filenum,filename="CosZenith", weights=weights)
#    plot_energy_zenith(Y_values=Y_labels,  Y_values_pred=Y_labels_pred,outdir=outdir,filenumber=filenum,filename="nocut")
#    plot_input_zenith(Y_labels_pred[:,0],X_DC, X_IC,outdir)
    if do_cuts:
      cutname="_E%s_%s"%(energy_min,energy_max)
      if args.cut_reco:
        cutname+="_cnnZenith%s_%s"%(zenith_min,zenith_max)
      if args.cut_true:
        cutname+="_trueZenith%s_%s"%(zenith_min,zenith_max)
      if args.cut_res:
        cutname+="_res_%s"%(zenith_res)
      if args.vertex_cut:
        cutname+="_"+args.vertex_cut
      plot_output_zenith(labels=Y_labels_cut, preds=Y_labels_pred_cut, outdir=outdir,filenumber=filenum,filename=cutname)
      plot_output(Y_labels, Y_labels_cut,outdir,filenumber=filenum,filename=cutname)
#      plot_charges_vs_energy(X_DC_cut, X_IC_cut,Y_labels_cut, outdir=outdir,filename=cutname)
#      plot_energy_zenith(Y_values=Y_labels_cut,  Y_values_pred=Y_labels_pred_cut,outdir=outdir,filenumber=filenum,filename=cutname)
      
#      plot_DOM_range(X_DC, Y_labels, X_DC_cut, Y_labels_cut,outdir,filenumber=filenum,filename="DC",cutname=cutname)
      '''
      plot_DOM_range(X_DC_cut, X_IC_cut,Y_labels,outdir,filenumber=filenum,filename="IC",cutname=cutname)

      plot_energy_zenith(Y_values=Y_labels_cut,  Y_values_pred=Y_labels_pred_cut,outdir=outdir,filenumber=filenum,filename=cutname)
      plot_output_zenith(Y_values=Y_labels_cut, Y_values_pred=Y_labels_pred_cut, outdir=outdir,filenumber=filenum,filename=cutname)
      plot_output(Y_labels_cut,outdir,filenumber=filenum,filename=cutname)
      '''
#        plot_energy_zenith(Y_values=Y_labels_cut,  Y_values_pred=Y_labels_pred_cut,outdir=outdir,filenumber=filenum,filename=cutname)
#      plot_output(Y_labels_cut,outdir,filenumber=filenum,filename=file_name)
#      plot_output_zenith(Y_labels_cut, Y_labels_pred_cut, outdir,filenumber=filenum,filename="Zenith")
    if reco_test is not None:
        plot_output(reco_labels,outdir,filenumber="%s_reco"%filenum)
if False: #do_input:
    print("Plotting inputs...")
#    plot_input_zenith(Y_labels[:,1],X_DC,X_IC,outdir,filename="True")
#    plot_input_zenith(Y_labels_pred[:,0],X_DC,X_IC,outdir,filename="Pred")

#    plot_input_strings(X_IC,X_IC_cut,outdir,filenumber=filenum,filename="IC",cutrange=file_name)
#    plot_input_strings(X_DC,X_DC_cut,outdir,filenumber=filenum,filename="DC", cutrange=file_name)
#    plot_DC_IC_strings(X_DC,X_DC_cut,X_IC,X_IC_cut,outdir,filenumber=filenum,filename="DC", cutrange=file_name)

    plot_input(X_DC,X_IC,outdir,filenumber=filenum,filename="nocut")
    plot_input(X_DC_cut,X_IC_cut,outdir,filenumber=filenum,filename=cutname)

