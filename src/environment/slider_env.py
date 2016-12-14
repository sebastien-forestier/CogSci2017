import matplotlib.pyplot as plt

from numpy import array, dot
from numpy.linalg import norm
from explauto.environment.environment import Environment
from explauto.utils import bounds_min_max


class SliderEnvironment(Environment):
    ''' Add a slider to an environment '''
    def __init__(self, env_cls, env_config, m_mins, m_maxs, s_mins, s_maxs, slider, combined_s):
        
        
        self.env = env_cls(**env_config)
        self.n_params_env = len(self.env.conf.m_dims)
        self.slider = slider
        self.combined_s = combined_s
        
        environment_seq = dict(m_mins = m_mins, 
                                    m_maxs = m_maxs, 
                                    s_mins = s_mins, 
                                    s_maxs = s_maxs)
        #print environment_seq
        Environment.__init__(self, **environment_seq)
        
    def rest_position(self):
        return self.env.rest_position
        
    def rest_params(self):
        return self.env.rest_params
    
    def compute_motor_command(self, m_ag):
        return self.env.compute_motor_command(m_ag)
        
    def compute_slider(self, s_env):
        slider_coord = array(self.slider['m_maxs']) - array(self.slider['m_mins'])
        pos = dot(array(s_env) - self.slider['m_mins'], slider_coord) / dot(slider_coord, slider_coord)
        
        dist = norm((self.slider['m_mins'] + pos*slider_coord) - array(s_env))
        #print "dist", dist
        if pos > self.slider['s_max'] or pos < self.slider['s_min'] or dist > self.slider['width']:
            pos = self.slider['default_s']
        return pos
    
    def trajectory(self, m): 
        return m
    
    def compute_sensori_effect(self, m):
        s_env = self.env.update(m)
        #print "slider s_env", self.env, s_env
#         print "s_env", s_env
#         print "s_slider", self.compute_slider(s_env)
#         print "s_comb", self.combined_s(s_env, self.compute_slider(s_env))
        s = self.combined_s(list(s_env), self.compute_slider(s_env))
        return bounds_min_max(s, self.conf.s_mins, self.conf.s_maxs)
    
    def plot(self, ax, m):
        if hasattr(self.env, 'plot_arm'):
            self.env.plot_arm(ax, m)
        else:
            self.env.plot(ax, m)
        ax.plot([self.slider['m_mins'][0], self.slider['m_maxs'][0]], [self.slider['m_mins'][1], self.slider['m_maxs'][1]], 'b')
        slider_coord = array(self.slider['m_maxs']) - array(self.slider['m_mins'])
        ort_vect = array([slider_coord[1], - slider_coord[0]]) / dot(slider_coord, slider_coord)
        ort_vect = self.slider['width'] * ort_vect / norm(ort_vect)
        #print 'ort_vect', ort_vect
        ax.plot([self.slider['m_mins'][0] + ort_vect[0], 
                 self.slider['m_maxs'][0] + ort_vect[0]], 
                [self.slider['m_mins'][1] + ort_vect[1], 
                 self.slider['m_maxs'][1] + ort_vect[1]], 'b')
        
        ax.plot([self.slider['m_mins'][0] - ort_vect[0], 
                 self.slider['m_maxs'][0] - ort_vect[0]], 
                [self.slider['m_mins'][1] - ort_vect[1], 
                 self.slider['m_maxs'][1] - ort_vect[1]], 'b')
        
        ax.plot([self.slider['m_mins'][0] + ort_vect[0], 
                 self.slider['m_mins'][0] - ort_vect[0]], 
                [self.slider['m_mins'][1] + ort_vect[1], 
                 self.slider['m_mins'][1] - ort_vect[1]], 'b')
        
        ax.plot([self.slider['m_maxs'][0] + ort_vect[0], 
                 self.slider['m_maxs'][0] - ort_vect[0]], 
                [self.slider['m_maxs'][1] + ort_vect[1], 
                 self.slider['m_maxs'][1] - ort_vect[1]], 'b')
        
        pos = self.compute_slider(self.env.update(m))
        pt = self.slider['m_mins'] + pos*slider_coord
        #print [pt[0]], [pt[1]]
        ax.scatter([pt[0]], [pt[1]])
    
            

if __name__ == '__main__':
    
    from explauto.environment.simple_arm import SimpleArmEnvironment
    
    armConfig = dict(
        m_mins = [-3, -3, -3],
        m_maxs = [3, 3, 3],
        s_mins = [-1,-1],
        s_maxs = [1, 1],
        length_ratio = 2,
        noise = 0
        )
    
    sConfig = dict(
                   env_cls = SimpleArmEnvironment,
                   env_config = armConfig, 
                   m_mins = armConfig['m_mins'], 
                   m_maxs = armConfig['m_maxs'], 
                   slider = dict(
                                 m_mins = [0., 1],
                                 m_maxs = [1., 0],
                                 s_min = 0,
                                 s_max = 1,
                                 default_s = -1,
                                 width = 0.1
                                 ), 
                   combined_s = lambda s, sl: s + [sl]
                   )
    
    sliderEnv = SliderEnvironment(**sConfig)
    
    m = [0.1,1.4,1]
    #m = [0,0,1]
    
    s =  sliderEnv.compute_sensori_effect(m)
    print "m=", m, "s=", s
    sliderEnv.plot(plt.subplot(), m)
    plt.show()