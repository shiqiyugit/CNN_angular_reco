# CNN_angular_reco

Branched off from https://github.com/jessimic/LowEnergyNeuralNetwork

Easier and cleaner to teach students for angular reconstruction. To retain history or more reco using FLERCNN, check out the original github 

Source environment first:
```source setup_combo_stable.sh```
Data preparation:
data_pre/i3_to_hdf5.py will convert i3 files to hdf5 keeping the variables needed and use cleaned pulses. 
data_pre/create_hdf5/ has some scripts ```.sh``` are for submitting to clusters where they will call ```.sb``` templates so that they submit jobs in batches using the same template
* data_pre/example/convert_i3_to_hdf5.sb gives an example of running i3 to hdf5 conversion using ```i3_to_hdf5.py``` interactively
* data_pre/example/flatten_zenith_for_hdf5.sb will apply flatten to the true distribution, zenith is the example here.
* * use --split so that it split all outcomes into a chosen number of hdf5 and get ready for training (will convert ```features``` into ```Y_train``` and ```Y_test``` with --split flag)
* python data_pre/example/apply_containmentcut.py will apply containment cut on the pre-processed training files so that different containment cuts can be quickly tested and no need to flattern again
* 

