import glob
import numpy
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_files",type=str,default=
#'flatten_NuMu_genie_149999_E0to3_CC_all_start_all_end_flat_101bins_9600evtperbin_file0?.hdf5',
#flatten_NuMu_genie_149999_E0to3_CC_all_start_all_end_flat_101bins_20000evtperbin_file0?.hdf5',
#'flatten_NuMu_genie_149999_E0to3_CC_all_start_all_end_flat_101bins_21400evtperbin_file0?.hdf5',
#'data_preflatten_NuMu_genie_149999_E0to3_CC_all_start_end_IC19_flat_101bins_26500evtperbin_file0?.hdf5',
#'NuMu_genie_149999_?_IC_level6.zst_cleanedpulses_transformed_IC78_E0to3_CC_all_start_all_end_flat_101bins_6550evtperbin.hdf5',
#'flatten_?_NuMu_genie_149999_level6.zst_cleanedpulses_transformed_IC19_correct_oscweights_E0to3_CC_all_start_all_end_flat_101bins_7000evtperbin.hdf5',
#'flatten_?_NuMu_genie_149999_level6.zst_cleanedpulses_transformed_IC19_E0to3_CC_contained_IC19_start_all_end_flat_101bins_6960evtperbin.hdf5',
#'flatten_?_NuMu_genie_149999_level6.zst_cleanedpulses_transformed_IC19_correct_oscweights_E0to3_CC_contained_IC19_start_all_end_flat_101bins_7000evtperbin.hdf5',
#'flatten_NuMu_genie_149999_E0to3_CC_all_start_all_end_flat_101bins_36170evtperbin_file0?.hdf5',
'NuMu_genie_149999_level6.zst_cleanedpulses_transformed_IC19_?_10_E0to3_CC_contained_IC19_start_contained_IC19_end_flat_101bins_6285evtperbin.hdf5',
                    dest="input_file", help="names for input files")
parser.add_argument("-d", "--path",type=str,default=
'/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/IC19startend/',
#'/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/AllIC/',
#'/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/',
#'/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/level6_hdf5_oldgcd/300G/allstartend/',
#'/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/IC19start_allend/final_oscweights/',
#'/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/IC19start_allend/final_right_oscwei/',
#'/mnt/research/IceCube/yushiqi2/Files/Flat_Zenith/flattening/allstartend/final_right_oscwei/',
                    dest="path", help="path to input files")
parser.add_argument("-n", "--name",type=str,default='output_with_train_300G',
                    dest="name", help="name for output file")
parser.add_argument("--old_reco",default=False,action='store_true',
                    dest="old_reco",help="use flag if concatonating all train, test, val into one file")
parser.add_argument("--save_train",type=bool,default=True, 
                    dest="save_train", help="save Y_train to plot weight of z")
args = parser.parse_args()

use_old_reco = args.old_reco
file_name_base = args.path + args.input_file
if not use_old_reco:
    file_names = sorted(glob.glob(file_name_base))
    print("Using %i files with names like %s"%(len(file_names), file_names[0]))
else:
    file_names = file_name_base
    print("Using file %s"%file_names)

name = args.name
if name is None:
    split_file_name = file_name_base[:-4]
    new_name = split_file_name[0]
    for name in range(1,len(split_file_name)-1):
        new_name = new_name + "_" + split_file_name[name]
    new_name += ".testonly.hdf5"
    output_file =  new_name
else:
    output_file = args.path + name + ".testonly.hdf5"

# Put all the test sets together
Y_test_use = None
X_test_DC_use = None
X_test_IC_use = None
Y_train_use = None
if use_old_reco:
    f = h5py.File(file_names, 'r')
    Y_test = f['Y_test'][:]
    X_test_DC = f['X_test_DC'][:]
    X_test_IC = f['X_test_IC'][:]
    weight_test=f['weights_test'][:]
    Y_train = f['Y_train'][:]
    X_train_DC = f['X_train_DC'][:]
    X_train_IC = f['X_train_IC'][:]
    weight_train=f['weights_train'][:]

    Y_validate = f['Y_validate'][:]
    X_validate_DC = f['X_validate_DC'][:]
    X_validate_IC = f['X_validate_IC'][:]
    weight_validate=f['weights_validate'][:]

    reco_test = f['reco_test'][:]
    reco_train = f['reco_train'][:]
    reco_validate = f['reco_validate'][:]
    f.close()
    del f

    print("Loaded all %i events"%(Y_test.shape[0]+Y_train.shape[0]+Y_validate.shape[0]))

    Y_test_use = numpy.concatenate((Y_test,Y_train,Y_validate))
    print("Concatted Y")
    del Y_test
    del Y_train
    del Y_validate
    X_test_DC_use = numpy.concatenate((X_test_DC,X_train_DC,X_validate_DC))
    print("Concatted DC")
    del X_test_DC
    del X_train_DC
    del X_validate_DC
    X_test_IC_use =numpy.concatenate((X_test_IC,X_train_IC,X_validate_IC))
    print("Concatted IC")
    del X_test_IC
    del X_train_IC
    del X_validate_IC
    reco_test_use = numpy.concatenate((reco_test,reco_train,reco_validate))
    del reco_test
    del reco_train
    del reco_validate
    print("Concatted reco")
    weight_test_use = numpy.concatenate((weight_test,weight_train,weight_validate))

else:
    for file in file_names:
        f = h5py.File(file, 'r')
        Y_test = f['Y_test'][:]
        X_test_DC = f['X_test_DC'][:]
        X_test_IC = f['X_test_IC'][:]

        if args.save_train:
          weight_test=f['weights_test'][:]
          Y_train = f['Y_train'][:]
#        reco_test = f['reco_test'][:]
        f.close()
        del f

        if Y_test_use is None:
            Y_test_use = Y_test
            X_test_DC_use = X_test_DC
            X_test_IC_use = X_test_IC
            Y_train_use = Y_train
            weight_test_use = weight_test
#            reco_test_use = reco_test
        else:
            Y_test_use = numpy.concatenate((Y_test_use, Y_test))
            X_test_DC_use = numpy.concatenate((X_test_DC_use, X_test_DC))
            X_test_IC_use = numpy.concatenate((X_test_IC_use, X_test_IC))
            if args.save_train:
              Y_train_use = numpy.concatenate((Y_train_use, Y_train))
              weight_test_use = numpy.concatenate((weight_test_use,weight_test))

#            reco_test_use = numpy.concatenate((reco_test_use, reco_test))

            del Y_test, X_test_DC, X_test_IC, weight_test
print(Y_test_use.shape)

print("Saving output file: %s"%output_file)
f = h5py.File(output_file, "w")
f.create_dataset("Y_test", data=Y_test_use)
f.create_dataset("X_test_DC", data=X_test_DC_use)
f.create_dataset("X_test_IC", data=X_test_IC_use)
if args.save_train:
  f.create_dataset("Y_train",data=Y_train_use)
  f.create_dataset("weight_test",data=weight_test_use)
#if use_old_reco:
#f.create_dataset("reco_test", data=reco_test_use)
f.close()
