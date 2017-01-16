import os
import sys
import cPickle
import numpy as np



# PARAMS
log_dir = "~/avakas/scratch/sforestier001/logs/CogSci2017/2017-01-15_18-30-39-TEST100"
config_list = ["RMB"]
n_iter = 100


    
filename = log_dir + '/results/vocal.pickle'
with open(filename, 'r') as f:
    data_vocal = cPickle.load(f)



trial_list = range(1,n_iter + 1) 

sound_tol = 0.4

for config_name in config_list:
    for trial in trial_list:
        
        data_vocal[config_name][trial]
        