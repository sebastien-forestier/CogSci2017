import sys
import numpy as np
import time

from explauto.utils import rand_bounds
sys.path.append('../')

from cogsci2017.environment.arm_diva_env import CogSci2017Environment


env = CogSci2017Environment(gui=False, audio=False)


n = 1000



t0 = time.time()

# Motor Babbling
for i in range(n):
    m = env.motor_babbling()
    #print "m", m
    s = env.update(m)
    #print "s", s
    
    
t1 = time.time()
    
print
print "Number of Arm trials:", env.count_arm
print "Number of Vocal trials:", env.count_diva
print "Number of Tool actions:", env.count_tool
print "Number of times Toy1 was reached by tool:", env.count_toy1_by_tool
print "Number of times Toy2 was reached by tool:", env.count_toy2_by_tool
print "Number of times Toy3 was reached by tool:", env.count_toy3_by_tool
print "Number of times Toy1 was reached by hand:", env.count_toy1_by_hand
print "Number of times Toy2 was reached by hand:", env.count_toy2_by_hand
print "Number of times Toy3 was reached by hand:", env.count_toy3_by_hand
print "Number of times parent gave vocal labels:", env.count_parent_give_label
print "Number of produced vowel /o/:", env.count_diva_o
print "Number of produced vowel /u/:", env.count_diva_u
print "Number of produced vowel /e/:", env.count_diva_e
print "Number of times parent gave object:", env.count_parent_give_object
print
print "Time for", n, "iterations:", t1 - t0, "sec"
print "Time per arm iteration", 1000.*env.time_arm_per_it, "ms"
print "Time per diva iteration", 1000.*env.time_diva_per_it, "ms"
