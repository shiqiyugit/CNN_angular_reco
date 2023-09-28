import matplotlib
#matplotlib.use('Agg')
import numpy
import argparse
import os
import glob
import matplotlib.pyplot as plt

from PlottingFunctions import plot_history_from_list
from PlottingFunctions import plot_history_from_list_split

parser = argparse.ArgumentParser()
#parser.add_argument("-i", "--input_folder",type=str,default="numu_flat_Z_5_150_CC_vertexDC_lrEpochs160_dropouts_0.2_custom_input_merge_v1",
#parser.add_argument("-i", "--input_folder",type=str,default="numu_flat_Z_5_150_CC_vertexDC_lrEpochs64_linear_3_5pi_flat_v1",
#parser.add_argument("-i", "--input_folder",type=str,default="linear_25_75pi_flat_lrEpochs64_lr0.00001_lrDrop0.9",
#parser.add_argument("-i", "--input_folder",type=str,default="numu_flat_Z_5_150_CC_vertexDC_lrEpochs64_linear_weighted_v1",
#parser.add_argument("-i", "--input_folder",type=str,default="numu_flat_Z_5_150_CC_vertexDC_lrEpochs128_linear_weighted_v0",
#parser.add_argument("-i", "--input_folder",type=str,default="lrEpochs64_lrInit0.001_lrDrop0.6_weighted",


#parser.add_argument("-i", "--input_folder",type=str,default='flat_zenith_all_start_all_end_flat_101bins_6600evtperbin_v0_lr0205',
parser.add_argument("-i", "--input_folder",type=str,default='flat_zenith_all_start_all_end_flat_101bins_11kevtperbin_v0',
#parser.add_argument("-i", "--input_folder",type=str,default='flat_zenith_all_start_all_end_flat_101bins_11000evtperbin_sigmoid',
#parser.add_argument("-i", "--input_folder",type=str,default='flat_zenith_all_start_all_end_flat_101bins_11kevtperbin_weighted',
#parser.add_argument("-i", "--input_folder",type=str,default='flat_zenith_all_start_all_end_flat_101bins_6600evtperbin_sigmoid',



#parser.add_argument("-i", "--input_folder",type=str,default="numu_flat_Z_5_150_CC_vertexDC_lrEpochs160_dropouts_0.2_custom_input_merge_v1",
#parser.add_argument("-i", "--input_folder",type=str,default="numu_flat_Z_5_150_CC_vertexDC_lrEpochs64_linear_3_5pi_flat_v1",
#parser.add_argument("-i", "--input_folder",type=str,default="linear_25_75pi_flat_lrEpochs64_lr0.00001_lrDrop0.9",
#parser.add_argument("-i", "--input_folder",type=str,default="numu_flat_Z_5_150_CC_vertexDC_lrEpochs64_linear_weighted_v1",
#parser.add_argument("-i", "--input_folder",type=str,default="numu_flat_Z_5_150_CC_vertexDC_lrEpochs128_linear_weighted_v0",
#parser.add_argument("-i", "--input_folder",type=str,default="lrEpochs64_lrInit0.001_lrDrop0.6_weighted",

#parser.add_argument("-i", "--input_folder",type=str,default="numu_flat_Z_5_150_CC_vertexDC_lrEpochs64_linear_weighted_v0",
#parser.add_argument("-i", "--input_folder",type=str,default="numu_flat_Z_5_150_CC_vertexDC_lrEpochs64_linear_pi_mirrored_v0_1",
#parser.add_argument("-i", "--input_folder",type=str,default="numu_flat_Z_5_150_CC_vertexDC_lrEpochs64_linear_25_75pi_weighted_v1",
#parser.add_argument("-i", "--input_folder",type=str,default="numu_flat_Z_5_150_CC_vertexDC_lrEpochs64_linear_25_75pi_v0",

#parser.add_argument("-i", "--input_folder",type=str,default="numu_flat_Z_5_150_CC_vertexDC_lrEpochs64_dropouts_0.2_custom_v1",
                    dest="input_folder", help="name of folder in output_plots")
parser.add_argument("-d","--dir",type=str,default="/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/output_plots",
                    dest="outplots_dir", help="path to output plots directory (including it)")
parser.add_argument("-e", "--epoch",default=None,
                    dest="epoch", help="number of epoch folder to grab from")
parser.add_argument("--ymin",default=0,
                    dest="ymin", help="min y value for loss plot")
parser.add_argument("--ymax",default=1,
                    dest="ymax", help="max y value for loss plot")
parser.add_argument("--title",default="",
                    dest='title',help="name of variable for title")
parser.add_argument("--best_epoch",default=None,
                    dest='best_epoch',help="value of epoch used for testing")
parser.add_argument("--variable",default="Zenith",
                    dest="variable",help="Name of variable for plotting")
args = parser.parse_args()

lr_start=0.001
lr_drop=0.58
#lr_epoch=64

plot_folder = args.input_folder
outplots_dir = args.outplots_dir
epoch = args.epoch
best_epoch = args.best_epoch
variable = args.variable
if best_epoch:
    best_epoch = int(best_epoch)
if epoch:
    epoch = int(epoch)

ymin = args.ymin
ymax = args.ymax
num_files = 7
start = 1 #start epoch for plotting
if ymin:
    ymin = float(ymin)
if ymax:
    ymax = float(ymax)

print("Epoch: %s, ymin: %s, ymax: %s"%(epoch,ymin,ymax))

full_path = "%s/%s/"%(outplots_dir,plot_folder)
file_name = "%ssaveloss_currentepoch.txt"%(full_path)
print("Using loss from %s"%file_name)

delimiter="\t"
f = open(file_name,"r")
header = f.readline().split(delimiter)
print(header)
header = header[:-1]
#header = numpy.delete(header, (4,8))
print(header)

rawdata = numpy.genfromtxt(file_name, skip_header=1)
'''
rawdata_1 = numpy.genfromtxt(file_name, skip_header=1, max_rows=64)
rawdata_1 = numpy.delete(rawdata_1, (4,8), 1)
print(rawdata_1[0])
rawdata_2 = numpy.genfromtxt(file_name, skip_header=1, usecols = range(0,7))
print(rawdata_2[0])
rawdata_2 = numpy.delete(rawdata_2, slice(0,64), 0)
print(rawdata_2[0])
rawdata=numpy.concatenate((rawdata_1, rawdata_2))
fout=open("newfile.txt","w")
numpy.savetxt(fout,rawdata)
fout.close()
quit()
'''
data = {}
for var in range(0,len(header)):
    data[header[var]] = rawdata[:epoch,var]

print(data.keys())

# Timing stats
plt.figure(figsize=(10,7))
plt.plot(data[header[0]],data[header[1]],label="load + train")
plt.plot(data[header[0]],data[header[2]],label="train")
plt.xlabel("epoch",fontsize=15)
plt.ylabel("times (minutes)",fontsize=15)
plt.title("Time To Train & Load",fontsize=25)
plt.legend()
savename = "TrainingTimePerEpoch"
if epoch:
    savename += "_%iEpochs"%epoch
plt.savefig("%s%s.png"%(full_path,savename))

# Loss Plots
loss=data['loss']
valloss=data['val_loss']
#valloss[valloss<0.001]=valloss[valloss<0.001]*1000000
#loss[loss<0.001]=loss[loss<0.001]*1000000
plot_history_from_list(loss, valloss,save=True, savefolder=full_path,logscale=False,variable=variable,pick_epoch=best_epoch,lr_start=lr_start,lr_drop=lr_drop, title=args.title, ymax=ymax, ymin=ymin)#,lr_epoch=lr_epoch)


# Average Validation Plot
def average_epochs(loss_list,start=None,end=None,file_num=7):
    
    save = 0
    avg_loss = []
    avg_epoch = []
    avg_range = []
    if not start:
        start = 1
    if not end:
        end = len(loss_list)+1
    
    for i in range(start,end):
        if save == 0:
            min_loss = loss_list[i-1]
            max_loss = loss_list[i-1]
        else:
            if min_loss > loss_list[i-1]:
                min_loss = loss_list[i-1]
            if max_loss < loss_list[i-1]:
                max_loss = loss_list[i-1]
        save += loss_list[i-1]
        if i%file_num == 0:
            avg_loss.append(save/file_num)
            avg_epoch.append(i)
            avg_range.append(max_loss-min_loss)
            save = 0
    
    min_full_pass = avg_loss.index(min(avg_loss))+1
    best_model = min_full_pass*file_num
    best_loss = min(avg_loss)
    print("Best loss at %i with value of %f"%(best_model,best_loss))
    return avg_loss, avg_epoch, best_model, best_loss, avg_range

avg_val_loss, avg_epoch,best_model,best_loss, avg_range = average_epochs(data['val_loss'],file_num=num_files)
name = header[-1][:-4]
avg_start = int(start/num_files)
if ymin is None:
    ymin=min(min(data['loss'][start:]),min(data['val_loss'][start:]))
if ymax is None:
    ymax=max(max(data['loss'][start:]),max(data['val_loss'][start:]))

#Plot Avg Loss
plt.figure(figsize=(10,7))
plt.plot([best_model,best_model],[ymin,ymax],linewidth=4,color='lime')
plt.plot(data["Epoch"][start:],data['loss'][start:],'b',label="%s Training"%name)
plt.plot(data["Epoch"][start:],data['val_loss'][start:],'c',label="%s Validation"%name)
plt.plot(avg_epoch[avg_start:],avg_val_loss[avg_start:],'r',label="Avg %s Validation"%name)
#plt.yscale('log')
plt.title("Training and Validation Loss after %s Epochs"%len(data['loss']),fontsize=25)
plt.xlabel('Epochs',fontsize=15)
plt.ylabel('Loss',fontsize=15)
plt.legend(loc="upper right",fontsize=15)
textstr = "Best Avg Model: %i \n Best Avg Loss: %.3f"%(best_model,best_loss)
props = dict(boxstyle='round', facecolor='blue', alpha=0.3)
plt.text(start+2,ymax-0.05*ymax, textstr, ha="left", va="center", fontsize=20,bbox=props)
savename = "AvgLossVsEpoch"
if epoch:
    savename += "_%iEpochs"%epoch
plt.savefig("%s%s.png"%(full_path,savename))

#Plot Range
plt.figure(figsize=(10,7))
plt.plot(avg_epoch[avg_start:],avg_range[avg_start:],'r.-')
#plt.yscale('log')
plt.xlabel('Epochs',fontsize=15)
plt.ylabel('Range of Loss per %i Epochs'%num_files,fontsize=15)
plt.title("Validation Loss Spread per Full Pass",fontsize=25)
savename = "AvgRangeVsEpoch"
if epoch:
    savename += "_%iEpochs"%epoch
plt.savefig("%s%s.png"%(full_path,savename))
