import os
import sys
import time
import cPickle
import numpy as np

sys.path.append('../')

from cogsci2017.environment.arm_diva_env import CogSci2017Environment
from cogsci2017.learning.supervisor import Supervisor
  


def run(log_dir, config_name, trial):
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    
    
    # PARAMS
    if config_name == "RMB":
        
        iterations = 80000
        model_babbling = "random"
        n_motor_babbling = 1000
        explo_noise = 0.05
        proba_imitate = 0.5
        gui=False
        audio=False
        
    elif config_name == "AMB":
        
        iterations = 80000
        model_babbling = "active"
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
    n_goals = 1000
    
    eval_results = {}
    
    for region in [1, 2, 3]:        
        eval_results[region] = {}
        for i in range(n_goals):
            eval_results[region][i] = {}
            environment.reset_toys(region=region)
            for toy in ["toy1", "toy2", "toy3"]:
                eval_results[region][i][toy] = {}
                                        
                if toy == "toy1":
                    goal = [environment.current_toy1[0] * (1. - t) / 2. for t in [0., 0.3, 0.5, 0.8, 1.]] + \
                           [environment.current_toy1[1] * (1. - t) / 2. for t in [0., 0.3, 0.5, 0.8, 1.]]
                    arm_mid = "mod3"
                    diva_mid = "mod10"
                elif toy == "toy2":
                    goal = [environment.current_toy2[0] * (1. - t) / 2. for t in [0., 0.3, 0.5, 0.8, 1.]] + \
                           [environment.current_toy2[1] * (1. - t) / 2. for t in [0., 0.3, 0.5, 0.8, 1.]]
                    arm_mid = "mod4"
                    diva_mid = "mod11"
                elif toy == "toy3":
                    goal = [environment.current_toy3[0] * (1. - t) / 2. for t in [0., 0.3, 0.5, 0.8, 1.]] + \
                           [environment.current_toy3[1] * (1. - t) / 2. for t in [0., 0.3, 0.5, 0.8, 1.]]
                    arm_mid = "mod5"
                    diva_mid = "mod12"
                    
                context = list(agent.modules[arm_mid].get_c(environment.get_current_context()))
                dists, _ = agent.modules[arm_mid].sm.model.imodel.fmodel.dataset.nn_y(context+goal)
                arm_dist = dists[0]
                
                if len(agent.modules[diva_mid].sm.model.imodel.fmodel.dataset) > 0:
                    context = list(agent.modules[diva_mid].get_c(environment.get_current_context()))
                    dists, _ = agent.modules[diva_mid].sm.model.imodel.fmodel.dataset.nn_y(context+goal)
                    diva_dist = dists[0]
                else:
                    diva_dist = np.inf
                
                if arm_dist < diva_dist:
                    m = agent.modules[arm_mid].inverse(np.array(context + goal), explore=False)            
                    m = list(m) + [0.]*28
                else:
                    m = agent.modules[diva_mid].inverse(np.array(context + goal), explore=False)            
                    m = [0.]*21 + list(m)
                    
                s = environment.update(m)
                
                if toy == "toy1":
                    reached = s[30:40]
                elif toy == "toy2":
                    reached = s[40:50]
                elif toy == "toy3":
                    reached = s[50:60]
                    
                comp_error = np.linalg.norm(np.array(reached) - np.array(goal))
                
                eval_results[region][i][toy]["goal"] = goal
                eval_results[region][i][toy]["reached"] = reached
                eval_results[region][i][toy]["tool"] = s[20:30]
                eval_results[region][i][toy]["comp_error"] = comp_error
                eval_results[region][i][toy]["arm_dist"] = arm_dist
                eval_results[region][i][toy]["diva_dist"] = diva_dist
        
        
    #print eval_results
    
    
    # DUMP LOG
    log = dict(environment=environment.save(),
               agent=agent.save(),
               eval_results=eval_results,
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
    