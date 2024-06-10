# CNN_angular_reco

Branched off from https://github.com/jessimic/LowEnergyNeuralNetwork

Easier and cleaner to teach students for angular reconstruction. To retain history or more reco using FLERCNN, check out the original github 

Singularity container has both icetray and tensorflow installed and can be used for training/testing can be found on cobalt: /data/ana/LE/oscNext/flercnn_sample_analysis/flercnn_reco/icetray_stable-tensorflow.sif

Source environment first:
```source setup_combo_stable.sh```
Data preparation:
data_pre/i3_to_hdf5.py will convert i3 files to hdf5 keeping the variables needed and use cleaned pulses. 
data_pre/create_hdf5/ has some scripts ```.sh``` are for submitting to clusters where they will call ```.sb``` templates so that they submit jobs in batches using the same template
* data_pre/example/convert_i3_to_hdf5.sb gives an example of running i3 to hdf5 conversion using ```i3_to_hdf5.py``` interactively
* data_pre/example/flatten_zenith_for_hdf5.sb will apply flatten to the true distribution, zenith is the example here.
* * use --split so that it split all outcomes into a chosen number of hdf5 and get ready for training (will convert ```features``` into ```Y_train``` and ```Y_test``` with --split flag)
* python data_pre/example/apply_containmentcut.py will apply containment cut on the pre-processed training files so that different containment cuts can be quickly tested and no need to flatten again
* CNN_Train_ic.py is the script for training. You can use singularity to train and test CNN on any machine has GPU
* ```CNN_angular_reco/condor_submit/``` has the example of run_training.sh to train the CNN and submit_condor.sub can be used to submit from sub-1 to npx to get a GPU node for training, which is requested. sub-1 can not access ```/data/``` area, so copy over entire ```condor_submit/``` to your home directory which is shared between ```/data/``` and npx
   * run ```ssh sub-1``` will get you to submission script, from there run ```condor_submit condor.sub ```. can check the status using ```condor_q``` or check the outcome file ```simple.err or simple.out```

Install environments locally using anaconda:
conda 4.10.1
python 3.8.5
conda env create -f conda_tfgpu_env.yml

