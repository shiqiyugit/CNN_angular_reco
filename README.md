# CNN_angular_reco

Branched off from https://github.com/jessimic/LowEnergyNeuralNetwork

Easier and cleaner to teach students for angular reconstruction. To retain history or more reco using FLERCNN, check out the original github 

Source environment first:
```source setup_combo_stable.sh```
Data preparation:
data_pre/i3_to_hdf5.py will convert i3 files to hdf5 keeping the variables needed and use cleaned pulses. 
data_pre/create_hdf5/ has some scripts ```.sh``` are for submitting to clusters where they will call ```.sb``` templates so that they submit jobs in batches using the same template
* data_pre/create_hdf5/example gives an example of running i3 to hdf5 conversion using ```i3_to_hdf5.py``` interactively
