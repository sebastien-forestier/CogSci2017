import sys
from explauto.utils import rand_bounds
sys.path.append('../')

from src.environment.arm_diva_env import CogSci2017Environment


env = CogSci2017Environment()


# Motor Babbling

for i in range(100):
    m = env.motor_babbling()
    #print "m", m
    s = env.update(m)
    #print "s", s