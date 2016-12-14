import numpy as np

from explauto.environment.simple_arm import SimpleArmEnvironment
from .slider_env import SliderEnvironment



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
s1Config = dict(
               env_cls = SimpleArmEnvironment,
               env_config = armConfig, 
               m_mins = armConfig['m_mins'], 
               m_maxs = armConfig['m_maxs'], 
               s_mins = [0], 
               s_maxs = [1], 
               slider = dict(
                             m_mins = [0.3, 0.3],
                             m_maxs = [0.3, -0.3],
                             s_min = 0,
                             s_max = 1,
                             default_s = 0,
                             width = 0.1
                             ), 
               combined_s = lambda s, sl: [sl]
               )


# Slider Arm
s1Config = dict(
               env_cls = SimpleArmEnvironment,
               env_config = armConfig, 
               m_mins = armConfig['m_mins'], 
               m_maxs = armConfig['m_maxs'], 
               s_mins = [0], 
               s_maxs = [1], 
               slider = dict(
                             m_mins = [0.5, 0.2],
                             m_maxs = [0.5, -0.2],
                             s_min = 0,
                             s_max = 1,
                             default_s = 0,
                             width = 0.15
                             ), 
               combined_s = lambda s, sl: [sl]
               )


# Slider Arm
s2Config = dict(
               env_cls = SliderEnvironment,
               env_config = s1Config, 
               m_mins = armConfig['m_mins'], 
               m_maxs = armConfig['m_maxs'], 
               s_mins = [0, 0], 
               s_maxs = [1, 1], 
               slider = dict(
                             m_mins = [-0.5, 0.2],
                             m_maxs = [-0.5, -0.2],
                             s_min = 0,
                             s_max = 1,
                             default_s = 0,
                             width = 0.15
                             ), 
               combined_s = lambda s, sl: s + [sl]
               )



n = 2

def combined_s(s):
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

    
#     if r3 < 0.25:
#         r4 = 0
#     elif r3 < 0.5:
#         r4 = r3
#     elif r3 < 0.75:
#         r4 =  r3 + np.random.normal(0,0.03)
#         if r4 > 1:
#             r4 = 1
#         if r3 < 0:
#             r4 = 0
#     else:
#         r4 = np.random.uniform(0.75,1)
#     return [r1, r2, r4]

    return [r1, r2, r3]


seqConfig = dict(
                   env_cls = SliderEnvironment,
                   env_config = s2Config,
                   environment = dict(m_mins = np.array(list(s2Config['m_mins'] * n)), 
                                    m_maxs = np.array(list(s2Config['m_maxs'] * n)), 
                                    s_mins = [0, 0, 0], 
                                    s_maxs = [1, 1, 1]),
                   m_mins = list(s2Config['m_mins'] * n),
                   m_maxs = list(s2Config['m_maxs'] * n),
                   env_seq_s_mins = [0, 0, 0],
                   env_seq_s_maxs = [1, 1, 1],
                   n = n,
                   combined_s = combined_s                 
                   )                 