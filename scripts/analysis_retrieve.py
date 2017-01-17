import os
import sys
import cPickle
import numpy as np



# PARAMS
log_dir = "/scratch/sforestier001/logs/CogSci2017/2017-01-16_21-05-23-COGSCI"
config_list = ["RMB", "AMB"]
n_iter = 10



# RETRIEVE LOGS
trial_list = range(1,n_iter + 1) 
data_vocal = {}
data_competence = {}

for config_name in config_list:
    data_vocal[config_name] = {}
    data_competence[config_name] = {}
    for trial in trial_list:
        print "Retrieve trial", trial
        data_vocal[config_name][trial] = {}
        data_competence[config_name][trial] = {}
        filename = log_dir + '/pickle/log-{}-{}'.format(config_name, trial) + '.pickle'
        with open(filename, 'r') as f:
            log = cPickle.load(f)
        f.close()
        
        # VOCAL
        data_vocal[config_name][trial]["errors"] = log["environment"]["best_vocal_errors_evolution"][::10]
        data_vocal[config_name][trial]["human_sounds"] = log["environment"]["human_sounds"]
        
        # COMPETENCE
        data_competence[config_name][trial]["eval_results"] = log["eval_results"]
        
        
# DUMP RESULT
if not os.path.exists(log_dir + "/results"):
    os.mkdir(log_dir + "/results")
    
filename = log_dir + '/results/vocal.pickle'
with open(filename, 'wb') as f:
    cPickle.dump(data_vocal, f)
f.close()
            
filename = log_dir + '/results/competence.pickle'
with open(filename, 'wb') as f:
    cPickle.dump(data_competence, f)
f.close()