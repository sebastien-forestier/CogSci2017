import os
import sys
import cPickle
import numpy as np



# PARAMS
log_dir = "/scratch/sforestier001/logs/CogSci2017/2017-01-17_19-32-17-EXPLO-0.5"
config_list = ["RMB", "AMB"]
n_iter = 500



# RETRIEVE LOGS
trial_list = range(1,n_iter + 1) 
data_vocal = {}
data_competence = {}
data_progress = {}

for config_name in config_list:
    data_vocal[config_name] = {}
    data_competence[config_name] = {}
    data_progress[config_name] = {}
    for trial in trial_list:
        print "Retrieve trial", trial
        data_vocal[config_name][trial] = {}
        data_competence[config_name][trial] = {}
        data_progress[config_name][trial] = {}
        filename = log_dir + '/pickle/log-{}-{}'.format(config_name, trial) + '.pickle'
        try:
            with open(filename, 'r') as f:
                log = cPickle.load(f)
            f.close()
            
            # VOCAL
            data_vocal[config_name][trial]["errors"] = log["environment"]["best_vocal_errors_evolution"]
            data_vocal[config_name][trial]["human_sounds"] = log["environment"]["human_sounds"]
            
            # COMPETENCE
            data_competence[config_name][trial]["eval_results"] = log["eval_results"]
            
            # PROGRESS
            data_progress[config_name][trial]["chosen_modules"] = log["agent"]["chosen_modules"]
            data_progress[config_name][trial]["cp_evolution"] = {}
            for mid in log["agent"]["cp_evolution"].keys():
                data_progress[config_name][trial]["cp_evolution"][mid] = log["agent"]["cp_evolution"][mid][::10]
            data_progress[config_name][trial]["pp_evolution"] = {}
            for mid in log["agent"]["pp_evolution"].keys():
                data_progress[config_name][trial]["pp_evolution"][mid] = log["agent"]["pp_evolution"][mid][::10]
                                      
            
        except IOError:
            print "Trial ", trial, "Not Found"
        
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
            
filename = log_dir + '/results/progress.pickle'
with open(filename, 'wb') as f:
    cPickle.dump(data_progress, f)
f.close()