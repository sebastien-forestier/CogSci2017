import numpy as np
from explauto.environment.environment import Environment
from explauto.utils import bounds_min_max


class MathEnvironment(Environment):
    ''' Mathematical environment '''
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs, f):
        
        self.f = f
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        
        
    def rest_position(self):
        return np.array([0.]*self.conf.m_dims)
        
    def rest_params(self):
        return self.rest_position
    
    def compute_motor_command(self, m_ag):
        return bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)
            
    def trajectory(self, m): 
        return m
    
    def compute_sensori_effect(self, m):
        s = self.f(m)
        return bounds_min_max(s, self.conf.s_mins, self.conf.s_maxs)    
            
# 
# m_mins = [0, 0, 0]
# m_maxs = [1, 1, 1]
# s_mins = [0, 0, 0, 0, 0, 0, 0]
# s_maxs = [1, 1, 1, 1, 1, 1, 1]

m_mins = [0]
m_maxs = [1]
s_mins = [0, 0]
s_maxs = [1, 1]

count = 0

def f(m):
    global count
    
    m1 = m[0]
    #m2 = m[1]
    #m3 = m[2]
    
    count += 1
    print "Environment ", count
    
    
    #s2 = np.abs((2*m2 - 1))
#     if count < 500:
#         s1 = m2
#     else:
    s1 = m1
        
    s2 = 2 * s1
#     
#     s3 = np.sqrt(s2)
#     
#     s4 = s3*s3
#     s3 = m3
#     s1 = m1*m1
#     #s2 = np.abs((2*m2 - 1))
#     s2 = m2
#     s3 = m3
#     #s3 = np.exp(-100*m3)
#     s4 = np.abs((s1*3 - 1)/2)
#     
#     if count < 5000:
#         s5 = (s2 + 2*s3)/3
#     else:
#         s5 = (2*s2 + s3)/3
#         
#     s6 = s4
#     #s6 = np.random.rand()
#     s7 = (s4+s5)/2
        
    
    return np.array([s1, s2])
    #return np.array([s1, s2, s3, s4, s5, s6, s7])

mathEnv_config = dict(m_mins=m_mins,
                      m_maxs=m_maxs,
                      s_mins=s_mins,
                      s_maxs=s_maxs,
                      f=f) 
