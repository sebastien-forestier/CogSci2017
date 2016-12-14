from numpy import log2, array

from explauto.environment.diva import DivaEnvironment
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

# sArm sDiva Combination
def combined_s(s):
    #print "comb s before", s
    c1 = s[0]
    c2 = s[1]
    c3 = 0
    
    ks = 0.5
    kd = 0.5
    if c1 > ks and c2 > kd:
        c3 = (1./(1-ks)) * (1./(1-kd)) * (c1-ks) * (c2-kd)
    return [c1, c2, c3]

combConfig = dict(
                   environment = dict(
                                          m_mins = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                                          m_maxs = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                          s_mins = [0, 0, 0.],
                                          s_maxs = [1, 1, 1.]
                                          ),
                   env1_cls = SliderEnvironment, 
                   env2_cls = SliderEnvironment,
                   env1_config = saConfig, 
                   env2_config = sdConfig,
                   combined_s = combined_s
                   )

        
        

#arm_diva_env = CombinedEnvironment(**combConfig)
# 
# m = [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# s = arm_diva_env.update(m)
# 
# print "m=", m, "s=", s