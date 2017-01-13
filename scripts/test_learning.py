import sys
import time
sys.path.append('../')

from cogsci2017.environment.arm_diva_env import CogSci2017Environment
from cogsci2017.learning.supervisor import Supervisor

iterations = 50000

environment = CogSci2017Environment(gui=False, audio=False)


config = dict(m_mins=environment.conf.m_mins,
             m_maxs=environment.conf.m_maxs,
             s_mins=environment.conf.s_mins,
             s_maxs=environment.conf.s_maxs)

agent = Supervisor(config, model_babbling="random", n_motor_babbling=1000, explo_noise=0.05, proba_imitate=0.5)

t0 = time.time()


count_social_tool_1 = 0
count_social_tool_2 = 0
count_social_tool_3 = 0


# Active Model Babbling
for i in range(iterations):
    if i % (iterations/10) == 0:
        print "Iteration", i
    context = environment.get_current_context()
    m = agent.produce(context)
    s = environment.update(m)
    agent.perceive(s)
    
    if environment.produced_sound:
        if agent.mid_control == "mod10": 
            count_social_tool_1 += 1
            print "----------------------------produced sound", environment.produced_sound, "while training to move toy1"
        elif agent.mid_control == "mod11": 
            count_social_tool_2 += 1
            print "----------------------------produced sound", environment.produced_sound, "while training to move toy2"
        elif agent.mid_control == "mod12": 
            count_social_tool_3 += 1
            print "----------------------------produced sound", environment.produced_sound, "while training to move toy3"
            
        
environment.print_stats()
agent.print_stats()

print 
print "# Social tool use to move toy 1:", count_social_tool_1
print "# Social tool use to move toy 2:", count_social_tool_2
print "# Social tool use to move toy 3:", count_social_tool_3
print

print "Time for", iterations, "iterations:", time.time() - t0, "sec"
print "Time per iteration", 1000*(time.time() - t0)/iterations, "ms"