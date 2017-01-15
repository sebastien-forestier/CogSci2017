import os
import sys
import time
import cPickle

sys.path.append('../')

from cogsci2017.environment.arm_diva_env import CogSci2017Environment
from cogsci2017.learning.supervisor import Supervisor
  


def run(log_dir, config_name, trial):
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    
    # PARAMS
    if config_name == "RMB":
        
        iterations = 20000
        model_babbling = "random"
        n_motor_babbling = 1000
        explo_noise = 0.05
        proba_imitate = 0.5
        gui=False
        audio=False
        
    else:
        raise NotImplementedError
    
    
    
    
    # INITIALIZE
    environment = CogSci2017Environment(gui=gui, audio=audio)
    
    config = dict(m_mins=environment.conf.m_mins,
                 m_maxs=environment.conf.m_maxs,
                 s_mins=environment.conf.s_mins,
                 s_maxs=environment.conf.s_maxs)
    
    agent = Supervisor(config, model_babbling=model_babbling, n_motor_babbling=n_motor_babbling, explo_noise=explo_noise, proba_imitate=proba_imitate)
    
    t0 = time.time()
    
    count_social_tool_1 = []
    count_social_tool_2 = []
    count_social_tool_3 = []
    count_social_tool_1_unmatched = []
    count_social_tool_2_unmatched = []
    count_social_tool_3_unmatched = []
    
    
    # LEARN
    for i in range(iterations):
        if i % (iterations/10) == 0:
            print "Iteration", i
        context = environment.get_current_context()
        m = agent.produce(context)
        s = environment.update(m)
        agent.perceive(s)
        
        if environment.produced_sound:
            if agent.mid_control == "mod10": 
                if environment.produced_sound == environment.human_sounds[0]:
                    count_social_tool_1 += [i]
                    print "++++++++++++++++++++++++++++produced sound MATCHED", environment.produced_sound, "while training to move toy1"
                else:
                    count_social_tool_1_unmatched += [i]
                    print "----------------------------produced sound NOT MATCHED", environment.produced_sound, "while training to move toy1"
            elif agent.mid_control == "mod11": 
                if environment.produced_sound == environment.human_sounds[1]:
                    count_social_tool_2 += [i]
                    print "++++++++++++++++++++++++++++produced sound MATCHED", environment.produced_sound, "while training to move toy2"
                else:
                    count_social_tool_2_unmatched += [i]
                    print "----------------------------produced sound NOT MATCHED", environment.produced_sound, "while training to move toy2"
            elif agent.mid_control == "mod12":  
                if environment.produced_sound == environment.human_sounds[2]:
                    count_social_tool_3 += [i]
                    print "++++++++++++++++++++++++++++produced sound MATCHED", environment.produced_sound, "while training to move toy3"
                else:
                    count_social_tool_3_unmatched += [i]
                    print "----------------------------produced sound NOT MATCHED", environment.produced_sound, "while training to move toy3"
                
            
    # PRINT STATS
    environment.print_stats()
    agent.print_stats()
    
    print 
    print "# Social tool use to move toy 1:", count_social_tool_1
    print "# Social tool use to move toy 1, unmatched:", count_social_tool_1_unmatched
    print "# Social tool use to move toy 2:", count_social_tool_2
    print "# Social tool use to move toy 2, unmatched:", count_social_tool_2_unmatched
    print "# Social tool use to move toy 3:", count_social_tool_3
    print "# Social tool use to move toy 3, unmatched:", count_social_tool_3_unmatched
    print
    
    print "Time for", iterations, "iterations:", time.time() - t0, "sec"
    print "Time per iteration", 1000*(time.time() - t0)/iterations, "ms"
    
    social_tool_use = dict(
                           count_social_tool_1=count_social_tool_1,
                           count_social_tool_1_unmatched=count_social_tool_1_unmatched,
                           count_social_tool_2=count_social_tool_2,
                           count_social_tool_2_unmatched=count_social_tool_2_unmatched,
                           count_social_tool_3=count_social_tool_3,
                           count_social_tool_3_unmatched=count_social_tool_3_unmatched)
    
    # ANALYSE COMPETENCE ERROR
    # TODO
    
    
    # DUMP LOG
    log = dict(environment=environment.save(),
               agent=agent.save(),
               social_tool_use=social_tool_use,)
    
    
    filename = log_dir + '/pickle/log-{}-{}'.format(config_name, trial) + '.pickle'
    with open(filename, 'wb') as f:
        cPickle.dump(log, f)
    f.close()
                
                

if __name__ == "__main__":
    
    log_dir = sys.argv[1]
    config_name = sys.argv[2]
    trial = sys.argv[3]
    
    run(log_dir, config_name, trial)
    