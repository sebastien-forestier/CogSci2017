import numpy as np

from numpy import array, hstack

from explauto.agent import Agent
from explauto.utils import rand_bounds
from explauto.utils.config import make_configuration
from explauto.exceptions import ExplautoBootstrapError

from sensorimotor_model import DemonstrableNN
from interest_model import MiscRandomInterest, ContextRandomInterest


class LearningModule(Agent):
    def __init__(self, mid, m_space, s_space, env_conf, explo_noise=0.1, context_mode=None):

        #print mid, m_space, s_space
        self.conf = make_configuration(env_conf.m_mins[m_space], 
                                       env_conf.m_maxs[m_space], 
                                       array(list(env_conf.m_mins) + list(env_conf.s_mins))[s_space],
                                       array(list(env_conf.m_maxs) + list(env_conf.s_maxs))[s_space])
        
        self.im_dims = self.conf.s_dims
        
        self.mid = mid
        self.m_space = m_space
        self.context_mode = context_mode
        self.s_space = s_space
        self.motor_babbling_n_iter = 0
        
        self.s = None
        self.sp = None
        self.last_interest = 0
        
        if context_mode is not None:
            im_cls, kwargs = (ContextRandomInterest, {
                               'win_size': 1000,
                               'competence_mode': 'knn',
                               'k': 20,
                               'progress_mode': 'local',
                               'context_mode':context_mode})
        else:
            im_cls, kwargs = (MiscRandomInterest, {
                               'win_size': 1000,
                               'competence_mode': 'knn',
                               'k': 20,
                               'progress_mode': 'local'})
            
        
        self.im = im_cls(self.conf, self.im_dims, **kwargs)
        
        sm_cls, kwargs = (DemonstrableNN, {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':explo_noise})
        self.sm = sm_cls(self.conf, **kwargs)
        
        Agent.__init__(self, self.conf, self.sm, self.im, context_mode=self.context_mode)
        

        
    def motor_babbling(self, n=1): 
        if n == 1:
            return rand_bounds(self.conf.m_bounds)[0]
        else:
            return rand_bounds(self.conf.m_bounds, n)
        
    def goal_babbling(self):
        s = rand_bounds(self.conf.s_bounds)[0]
        m = self.sm.infer(self.conf.s_dims, self.conf.m_dims, s)
        return m
            
    def get_m(self, ms): return array(ms[self.m_space])
    def get_s(self, ms): return array(ms[self.s_space])
        
    def set_one_m(self, ms, m):
        """ Set motor dimensions used by module
        """
        ms = array(ms)
        ms[self.mconf['m']] = m
        
    def set_m(self, ms, m):
        """ Set motor dimensions used by module on one ms
        """
        self.set_one_m(ms, m)
        if self.mconf['operator'] == "seq":
            return [array(ms), array(ms)]
        elif self.mconf['operator'] == "par":
            return ms
        else:
            raise NotImplementedError
    
    def set_s(self, ms, s):
        """ Set sensory dimensions used by module
        """
        ms = array(ms)
        ms[self.mconf['s']] = s
        return ms          
    
    def inverse(self, s, explore=False):
        self.m,_ = self.infer(self.conf.s_dims, self.conf.m_dims, s, pref='', explore=explore)
        return self.m
        
    def infer(self, expl_dims, inf_dims, x, pref='', explore=True):      
        mode = "explore" if explore else "exploit"
        self.sensorimotor_model.mode = mode
        m, sp = self.sensorimotor_model.infer(expl_dims, inf_dims, x.flatten())
        return m, sp
            
    def produce(self, context=None, j_sm=None):
        if self.t < self.motor_babbling_n_iter:
            self.m = self.motor_babbling()
            self.s = np.zeros(len(self.s_space))
            self.x = np.zeros(len(self.expl_dims))
        else:
            self.x = self.choose(context)
            self.y, self.sp = self.infer(self.expl_dims, self.inf_dims, self.x)
            
            self.m, self.s = self.extract_ms(self.x, self.y)
                   
        return self.m        
    
    def update_sm(self, m, s): 
        self.sensorimotor_model.update(m, s)   
        self.t += 1 
    
    def update_im(self, m, s):
        if self.t >= self.motor_babbling_n_iter:
            self.interest_model.update(hstack((m, self.s)), hstack((m, s)), self.sp)
        
    def competence(self): return self.interest_model.competence()
    def progress(self): return self.interest_model.progress()
    def interest(self): return self.interest_model.interest()

    def perceive(self, m, s):
        self.update_sm(m, s)
        self.update_im(m, s)
