import numpy as np

from numpy import log2, array

from explauto.environment.diva import DivaEnvironment
from explauto.environment.simple_arm import SimpleArmEnvironment
from .slider_env import SliderEnvironment
from .combined_env import CombinedEnvironment


# Arm
armConfig = dict(
    m_mins = [-1, -1, -1],
    m_maxs = [1, 1, 1],
    s_mins = [-1,-1],
    s_maxs = [1, 1],
    length_ratio = 2,
    noise = 0,
    unit = 'deg'
    )

# Slider Arm
saConfig = dict(
               env_cls = SimpleArmEnvironment,
               env_config = armConfig, 
               m_mins = armConfig['m_mins'], 
               m_maxs = armConfig['m_maxs'], 
               s_mins = [0], 
               s_maxs = [1], 
               slider = dict(
                             m_mins = [0., 1.],
                             m_maxs = [0., -1.],
                             s_min = 0,
                             s_max = 1,
                             default_s = 0,
                             width = 0.1
                             ), 
               combined_s = lambda s, sl: [sl]
               )

#sliderEnv = SliderEnvironment(**saConfig)
    
# Diva
sound_o = list(log2([500, 900]))
sound_y = list(log2([300, 1700]))
sound_u = list(log2([300, 800]))
sound_e = list(log2([600, 1700]))

divaConfig = dict(
                m_mins = array([-1, -1, -1, -1, -1, -1, -1]),
                m_maxs = array([1, 1, 1, 1, 1, 1, 1]),
                s_mins = array([ 7.,  9.]),
                s_maxs = array([ 10. ,  11.5]),
                m_used = range(7),
                s_used = range(1, 3),
                rest_position_diva = list([0]*7),
                audio = False,
                diva_use_initial = False,
                diva_use_goal = True,
                used_diva = list([True]*7),
                n_dmps_diva = 7,
                n_bfs_diva = 0,
                move_steps = 10,
                )
        

# Slider Diva
sdConfig = dict(
               env_cls = DivaEnvironment,
               env_config = divaConfig, 
               m_mins = divaConfig['m_mins'], 
               m_maxs = divaConfig['m_maxs'], 
               s_mins = [0], 
               s_maxs = [1], 
               slider = dict(
                             m_mins = sound_o,
                             m_maxs = sound_y,
                             s_min = 0,
                             s_max = 1,
                             default_s = 0,
                             width = 0.3
                             ), 
               combined_s = lambda s, sl: [sl]
               )


combConfig = dict(
                   environment = dict(
                                          m_mins = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                          m_maxs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                          s_mins = [0, 0],
                                          s_maxs = [1, 1]
                                          ),
                   env1_cls = SliderEnvironment, 
                   env2_cls = SliderEnvironment,
                   env1_config = saConfig, 
                   env2_config = sdConfig,
                   combined_s = lambda s:s
                   )

        

n = 2

def combined_s2(s):
    #print "comb s before", s
    s11 = s[0][0]
    s21 = s[0][1]
    s12 = s[1][0]
    s22 = s[1][1]
    
    r1 = (s11 + s21) / 2.    
    r2 = (s12 + s22) / 2.
        
    if r1 and r2:
        r3 = r1 + r2
    else:
        r3 = 0.
#         
#     if (s11 and s22) or (s21 and s12):
#         r = (s11 + s12 + s21 + s22) / 2.
#     elif s11 and s12:
#         r = s12 / 2.
#     elif s21 and s22:
#         r = s22 / 2.
#     else:
#         r = (s11 + s12 + s21 + s22) / 2.
#     
    return [r1, r2, r3]


seqConfig = dict(
                   env_cls = CombinedEnvironment,
                   env_config = combConfig,
                   environment = dict(m_mins = np.array(list(combConfig['environment']['m_mins'] * n)), 
                                    m_maxs = np.array(list(combConfig['environment']['m_maxs'] * n)), 
                                    s_mins = [0, 0, 0], 
                                    s_maxs = [1, 1, 1]),
                   m_mins = list(combConfig['environment']['m_mins'] * n),
                   m_maxs = list(combConfig['environment']['m_maxs'] * n),
                   env_seq_s_mins = [0, 0, 0],
                   env_seq_s_maxs = [1, 1, 1],
                   n = n,
                   combined_s = combined_s2             
                   )                 
        

#arm_diva_env = CombinedEnvironment(**combConfig)
# 
# m = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# s = arm_diva_env.update(m)
# 
# print "m=", m, "s=", s