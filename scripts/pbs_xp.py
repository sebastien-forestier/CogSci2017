import os
import sys
import subprocess
import time
import datetime
import cPickle
import json
import base64
import numpy as np


# CONFIGS
config_list = ["RMB"]

n_iter = 10


path = '/home/sforestier/software/CogSci2017/scripts/'
start_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
pool_name = sys.argv[1]


def write_pbs(config_name, trial, log_dir):
    pbs =   """
#!/bin/sh

#PBS -l walltime=01:00:00
#PBS -l nodes=1:ppn=1
#PBS -N {}-{}
#PBS -o {}logs/log-{}-{}.output
#PBS -e {}logs/log-{}-{}.error

cd {}
time python run.py {} {} {}

""".format(config_name, trial, log_dir, config_name, trial, log_dir, config_name, trial, path, log_dir, config_name, trial)
    filename = '{}-{}.pbs'.format(config_name, trial)
    with open(log_dir + "pbs/" + filename, 'wb') as f:
        f.write(pbs)


log_dir = '/scratch/sforestier001/logs/CogSci2017/' + start_date + '-' + pool_name



trial_list = range(1,n_iter + 1) 


os.mkdir(log_dir + "/")
os.mkdir(log_dir + "/" + "pbs")
os.mkdir(log_dir + "/" + "img")
os.mkdir(log_dir + "/" + "logs")
#os.mkdir(log_dir + "/" + "configs")
    
    
    
for trial in trial_list:
    for config_name in config_list:
        print config_name, trial
        write_pbs(config_name, trial, log_dir + "/")   
        filename = '{}-{}.pbs'.format(config_name, trial)
        print "qsub " + log_dir + "/pbs/" + filename
        process = subprocess.Popen("qsub " + log_dir + "/pbs/" + filename, shell=True, stdout=subprocess.PIPE)
        time.sleep(0.2)
