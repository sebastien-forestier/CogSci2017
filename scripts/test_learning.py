import sys
import time
sys.path.append('../')

from cogsci2017.environment.arm_diva_env import CogSci2017Environment
from cogsci2017.learning.supervisor import Supervisor

iterations = 10000

environment = CogSci2017Environment(gui=False, audio=False)


config = dict(m_mins=environment.conf.m_mins,
             m_maxs=environment.conf.m_maxs,
             s_mins=environment.conf.s_mins,
             s_maxs=environment.conf.s_maxs)

agent = Supervisor(config)

t0 = time.time()

# Active Model Babbling
for i in range(iterations):
    context = environment.get_current_context()
    m = agent.produce(context)
    s = environment.update(m)
    agent.perceive(s)
    
environment.print_stats()
agent.print_stats()

print "Time for", iterations, "iterations:", time.time() - t0, "sec"
print "Time per iteration", 1000*(time.time() - t0)/iterations, "ms"