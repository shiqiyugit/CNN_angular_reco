#########################
# Version of CNN on 8 Nov 2019
# 
# Runs net and plots
# Takes in multiple files to train on (large dataset)
# Runs Energy and Zenith only (must do both atm)
####################################

import numpy as np
import h5py
import time
import math
import os, sys
import argparse
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",type=str,default='flatten_NuMu_genie_149999_level6.zst_cleanedpulses_transformed_IC78_0_10_E0to3_CC_all_start_all_end_flat_101bins_7750evtperbin.hdf5',
                    dest="input_files", help="name and path for input files")
parser.add_argument("-d", "--path",type=str,default='/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/AllIC/',
                    dest="path", help="path to input files")
parser.add_argument("-o", "--output_dir",type=str,default='/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork',
                    dest="output_dir", help="path to output_plots directory, do not end in /")
parser.add_argument("-n", "--name",type=str,default='test',
                    dest="name", help="name for output directory and model file name")
parser.add_argument("-e","--epochs", type=int,default=10,
                    dest="epochs", help="number of epochs for neural net")
parser.add_argument("--start", type=int,default=None,
                    dest="start_epoch", help="epoch number to start at")
parser.add_argument("--sub_dropout", type=float,default=0.2,
                        dest='sub_dropout',help="ID dropout")
parser.add_argument("--dropout", type=float,default=0.2,
                        dest='dropout', help="merged dropout")
parser.add_argument("--variables", type=int,default=1,
                    dest="train_variables", help="1 for [energy], 2 for [energy, zenith], 3 for [energy, zenith, track]")

parser.add_argument("--network",type=str,default='make_network',
                    dest="network",help="Name of network version that has make_network to setup network configuration")

parser.add_argument("--model",type=str, default="",
                    dest="model",help="Name of file with model weights to load--will start from here if file given")

parser.add_argument("--no_test",type=str,default=True,
                    dest="no_test",help="Don't do any testing")
parser.add_argument("--energy_loss", type=float,default=1,
                    dest="energy_loss", help="factor to divide energy loss by")
parser.add_argument("--first_variable", type=str,default="zenith",
                    dest="first_variable", help = "name for first variable (energy, zenith only two supported)")
parser.add_argument("--lr", type=float,default=0.0001,
                    dest="learning_rate",help="learning rate as a FLOAT")
parser.add_argument("--lr_drop", type=float,default=0.58,
                    dest="lr_drop",help="factor to drop learning rate by each lr_epoch")
parser.add_argument("--lr_epoch", type=int,default=64,
                    dest="lr_epoch",help="step size for number of epochs LR changes")
parser.add_argument("--batch_size", type=int,default=128,
                    dest="batch_size",help="batch size")
parser.add_argument("--zenith_min", type=float,default=0.,
                    dest="zmin",help="cut on training sample, min cos(zenith) value")
parser.add_argument("--zenith_max", type=float,default=3.14,
                    dest="zmax",help="cut on training sample, max cos(zenith) value")
parser.add_argument("--weighted_loss", type=bool, default=False,
                    dest="weighted_loss", help="weighted loss to enhance boundary loss values")
parser.add_argument("--weights", type=str, default=None, #'/mnt/home/yushiqi2/Analysis/LowEnergyNeuralNetwork/utils/weight/spectral_weight.txt',
                    dest="weights", help="txt file saving zenith weight")
parser.add_argument("--oscweight", type=bool, default=False,
                    dest="oscweight", help="use oscweight")
parser.add_argument("--auto_train", type=bool, default=False,
                    dest="auto_train", help="auto train using the last model")
parser.add_argument("--activation", default='linear',
                    dest="activation", help="activation function")

args = parser.parse_args()

# Settings from args
input_files = args.input_files
path = args.path
num_epochs = args.epochs
filename = args.name
network = args.network

epochs_step_drop = args.lr_epoch
if epochs_step_drop is None or epochs_step_drop==0:
    no_change = 0
    lr_drop = 1
    epochs_step_drop = 1
    # set up so pow(lr_drop,(epoch+1)/epochs_step_drop*nochange) = 1
    print("YOU DIDN'T GIVE A STEP SIZE TO DROP LR, WILL NOT CHANGE LR")
else:
    no_change = 1
    lr_drop = args.lr_drop

initial_lr = args.learning_rate

train_variables = args.train_variables
batch_size = args.batch_size
sub_dropout = args.sub_dropout
DC_drop_value = sub_dropout
IC_drop_value = sub_dropout
connected_drop_value = args.dropout
zenith_min = args.zmin
zenith_max = args.zmax
if args.zmax==3.14:
  zenith_max = math.pi
start_epoch = args.start_epoch
energy_loss_factor = args.energy_loss

old_model_given = args.model
    
if args.no_test == "true" or args.no_test =="True":
    no_test = True
else:
    no_test = False

if args.first_variable == "Zenith" or args.first_variable == "zenith" or args.first_variable == "Z" or args.first_variable == "z":
    first_var = "zenith"
    first_var_index = 1
    print("Assuming Zenith is the only variable to train for")
    assert train_variables==1,"DOES NOT SUPPORT ZENITH FIRST + additional variables"
elif args.first_variable == "energy" or args.first_variable == "energy" or args.first_variable == "e" or args.first_variable == "E":
    first_var = "energy"
    first_var_index = 0
    print("training with energy as the first index")
else:
    first_var = "energy"
    first_var_index = 0
    print("only supports energy and zenith right now! Please choose one of those. Defaulting to energy")
    print("training with energy as the first index")

save = True
save_folder_name = "%s/output_plots"%(args.output_dir)

if save==True:
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)
    save_folder_name = "%s/output_plots/%s/"%(args.output_dir,filename)
    if os.path.isdir(save_folder_name) != True:
        os.mkdir(save_folder_name)
make_header_saveloss = False
if os.path.isfile("%ssaveloss_currentepoch.txt"%(save_folder_name)) != True:
    make_header_saveloss = True

if args.auto_train:
  import re
  start_epoch=0
  old_model_given=None
  models = sorted(glob.glob("%s/output_plots/%s/*epochs_model.hdf5"%(args.output_dir,filename)))
  for model in models: 
    temp = re.findall(r'\d+', model)
    res = int(list(temp)[-2])
    if res>start_epoch:
      start_epoch=res
      old_model_given=model
  print("auto train using old model: ", old_model_given)
use_old_reco = False

files_with_paths = os.path.join(path,input_files)
file_names = sorted(glob.glob(files_with_paths))
print("\nFiles Used \nTraining %i files that look like %s \nStarting with model: %s \nSaving output to: %s"%(len(file_names),file_names[0],old_model_given,save_folder_name))

print("\nNetwork Parameters \nbatch_size: %i \nsub_dropout: %f \nstarting learning rate: %f \n"%(batch_size,sub_dropout,initial_lr))

print("Starting at epoch: %s \nTraining until: %s epochs \nTraining on %s variables with %s first"%(start_epoch,start_epoch+num_epochs,train_variables, first_var))

afile = file_names[0]
f = h5py.File(afile, 'r')
X_train_DC = f['X_train_DC'][:]
X_train_IC1 = f['X_train_IC1'][:]
X_train_IC2 = f['X_train_IC2'][:]
X_train_IC3 = f['X_train_IC3'][:]

f.close()
del f
print("Train Data DC", X_train_DC.shape)
print("Train Data IC", X_train_IC1.shape)

# LOAD MODEL
from utils.cnn_model import make_network, make_network_DC, make_network_3D, scaled_sigmoid
from utils.mobilenetv2 import MobileNetV2

activation=args.activation
if args.activation == "scaled_sigmoid":
  activation=scaled_sigmoid

if network == "make_network":
    model_DC = make_network(X_train_DC,X_train_IC,train_variables,DC_drop_value,IC_drop_value,connected_drop_value, activation)
elif network == "make_network_DC":
    model_DC = make_network_DC(X_train_DC, train_variables,DC_drop_value,IC_drop_value,connected_drop_value)
elif network == "make_network_3D":
    model_DC = make_network_3D(X_train_DC,X_train_IC1, X_train_IC2, X_train_IC3, train_variables,DC_drop_value,IC_drop_value,connected_drop_value)
elif network == "make_mobilenetv2":
    model_DC = MobileNetV2(X_train)
else:
    print("PICK A NETWORK TO TRAIN")
    quit()

del X_train_DC,X_train_IC1, X_train_IC2, X_train_IC3

# WRITE OWN LOSS FOR MORE THAN ONE REGRESSION OUTPUT
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.losses import mean_squared_error, mean_absolute_error, MeanAbsoluteError, mean_absolute_percentage_error

def ZenithLoss(y_truth,y_predicted):
    return mean_absolute_error(y_truth[:,1],y_predicted[:,0])

def EnergyLoss(y_truth,y_predicted):
    #return mean_absolute_percentage_error(y_truth[:,0],y_predicted[:,0])
    return mean_absolute_error(y_truth[:,0],y_predicted[:,0])

# Run neural network and record time ##

if os.path.isdir(save_folder_name+"logs") != True:
      os.mkdir(save_folder_name+"logs")
#log_dir = save_folder_name+"logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")    
log_dir = save_folder_name+"logs/"
tensorboard_callback = TensorBoard(log_dir=log_dir, write_images=True)

end_epoch = start_epoch + num_epochs
t0 = time.time()

for epoch in range(start_epoch,end_epoch):
   
    learning_rate = initial_lr * math.pow(lr_drop, math.floor((1+epoch)/epochs_step_drop)*no_change) 
    print(initial_lr, lr_drop, epoch, epochs_step_drop, no_change)
        
    ## NEED TO HAVE OUTPUT DATA ALREADY TRANSFORMED!!! ##
    #print("True Epoch %i/%i"%(epoch+1,num_epochs))
    t0_loading = time.time()
    # Get new dataset
    input_file = file_names[epoch%len(file_names)]
    print("Now using file %s with lr %.1E"%(input_file,learning_rate))
    f = h5py.File(input_file, 'r')
    Y_train = f['Y_train'][:]
    X_train_DC = f['X_train_DC'][:]
    X_train_IC1 = f['X_train_IC1'][:]
    X_train_IC2 = f['X_train_IC2'][:]
    X_train_IC3 = f['X_train_IC3'][:]

    print("done converting training set")

    z_train_weight = None
    z_validate_weight = None

    X_validate_DC = f['X_validate_DC'][:]
    X_validate_IC1 = f['X_validate_IC1'][:]
    X_validate_IC2 = f['X_validate_IC2'][:]
    X_validate_IC3 = f['X_validate_IC3'][:]

    Y_validate = f['Y_validate'][:]

    print("Shape of sample",Y_train.shape)

    f.close()
    del f
    print("compile the model...")

    if first_var == "zenith":
      model_DC.compile(loss=ZenithLoss,
            optimizer=Adam(lr=learning_rate),
            metrics=[ZenithLoss])
    else:
      model_DC.compile(loss=EnergyLoss,
            optimizer=Adam(lr=learning_rate),
            metrics=[EnergyLoss])

    # Compile model

	# Use old weights
    if epoch > 0 and old_model_given is None:
        print("Using current model")
        last_model = '%scurrent_model_while_running.hdf5'%save_folder_name
        model_DC.load_weights(last_model)
    elif old_model_given:
        print("Using given model %s"%old_model_given)
        model_DC.load_weights(old_model_given)
        old_model_given = None
    else:
        print("Training set: %i, Validation set: %i"%(len(Y_train),len(Y_validate)))
        print(epoch,end_epoch)
    
    #Run one epoch with dataset
    t0_epoch = time.time()
    print("strat training...")
    if network == "make_network_DC":
          network_history = model_DC.fit([X_train_DC], Y_train, sample_weight=z_train_weight,
                            validation_data= ([X_validate_DC], Y_validate, z_validate_weight),
                            batch_size=batch_size,
                            initial_epoch= epoch,
                            epochs=epoch+1, #goes from intial to epochs, so need it to be greater than initial
                            callbacks = [ModelCheckpoint('%scurrent_model_while_running.hdf5'%save_folder_name),tensorboard_callback],
                            verbose=1)
    else:
          network_history = model_DC.fit([X_train_DC, X_train_IC1, X_train_IC2, X_train_IC3], Y_train, sample_weight=z_train_weight,
                            validation_data= ([X_validate_DC, X_validate_IC1, X_validate_IC2, X_validate_IC3], Y_validate, z_validate_weight),
                            batch_size=batch_size,
                            initial_epoch= epoch,
                            epochs=epoch+1, #goes from intial to epochs, so need it to be greater than initial
                            callbacks = [ModelCheckpoint('%scurrent_model_while_running.hdf5'%save_folder_name),tensorboard_callback],
                            verbose=1)
    t1_epoch = time.time()
    t1_loading = time.time()
    dt_epoch = (t1_epoch - t0_epoch)/60.
    dt_loading = (t1_loading - t0_loading)/60.
   
    #Set up file that saves losses once
    if make_header_saveloss and epoch==0:
        afile = open("%ssaveloss_currentepoch.txt"%(save_folder_name),"a")
        afile.write("Epoch" + '\t' + "Time Epoch" + '\t' + "Time Train" + '\t')
        for key in network_history.history.keys():
            afile.write(str(key) + '\t')
        afile.write('\n')
        afile.close()
    # Save loss
    afile = open("%ssaveloss_currentepoch.txt"%(save_folder_name),"a")
    afile.write(str(epoch+1) + '\t' + str(dt_loading) + '\t' + str(dt_epoch) + '\t')
    for key in network_history.history.keys():
        afile.write(str(network_history.history[key][0]) + '\t')

    afile.write('\n')    
    afile.close()
    del afile
    print(epoch,len(file_names),epoch%len(file_names),len(file_names)-1)
    # save at the last epoch
    if epoch%len(file_names) == (len(file_names)-1):
      model_save_name = "%s%s_%iepochs_model.hdf5"%(save_folder_name,filename,epoch+1)
      model_DC.save(model_save_name)
      print("Saved model to %s"%model_save_name)
      
    del X_train_DC, X_train_IC1, X_train_IC2, X_train_IC3, X_validate_DC, X_validate_IC1, X_validate_IC2, X_validate_IC3
    del Y_train, Y_validate, z_train_weight

t1 = time.time()
print("This took me %f minutes"%((t1-t0)/60.))

model_DC.save("%s%s_model_final.hdf5"%(save_folder_name,filename))
del model_DC
sys.exit()
