import numpy as np

from explauto.utils import rand_bounds, bounds_min_max, softmax_choice, prop_choice
from explauto.utils.config import make_configuration
from learning_module import LearningModule


class Supervisor(object):
    def __init__(self, config, model_babbling="random", n_motor_babbling=0, explo_noise=0.1, choice_eps=0.2):
        
        self.config = config
        self.model_babbling = model_babbling
        self.n_motor_babbling = n_motor_babbling
        self.explo_noise = explo_noise
        self.choice_eps = choice_eps
        
        self.conf = make_configuration(**config)
        
        self.t = 0
        self.modules = {}
        self.chosen_modules = []
        self.progresses_evolution = {}
        self.interests_evolution = {}
        
        self.mid_control = None
        
        # Define motor and sensory spaces:
        m_ndims = self.conf.m_ndims # number of motor parameters
        
        self.arm_n_dims = 21
        self.diva_n_dims = 28
        
        self.m_arm = range(self.arm_n_dims)
        self.m_diva = range(self.arm_n_dims,self.arm_n_dims + self.diva_n_dims)
        self.m_space = range(m_ndims)
        self.c_dims = range(m_ndims, m_ndims+10)
        self.s_hand = range(m_ndims+10, m_ndims+20)
        self.s_tool = range(m_ndims+20, m_ndims+30)
        self.s_toy1 = range(m_ndims+30, m_ndims+40)
        self.s_toy2 = range(m_ndims+40, m_ndims+50)
        self.s_toy3 = range(m_ndims+50, m_ndims+60)
        self.s_sound = range(m_ndims+60, m_ndims+70)
        self.s_caregiver = range(m_ndims+70, m_ndims+80)
        
        self.s_spaces = dict(s_hand=self.s_hand, 
                             s_tool=self.s_tool, 
                             s_toy1=self.s_toy1, 
                             s_toy2=self.s_toy2, 
                             s_toy3=self.s_toy3, 
                             s_sound=self.s_sound, 
                             s_caregiver=self.s_caregiver)
        
        # Create the 10 learning modules:
        #self.modules['mod1'] = LearningModule("mod1", self.m_arm, self.c_dims + self.s_hand, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)
        #self.modules['mod2'] = LearningModule("mod2", self.m_arm, self.c_dims + self.s_tool, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)
        #self.modules['mod3'] = LearningModule("mod3", self.m_arm, self.c_dims + self.s_toy1, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)
        #self.modules['mod4'] = LearningModule("mod4", self.m_arm, self.c_dims + self.s_toy2, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)
        #self.modules['mod5'] = LearningModule("mod5", self.m_arm, self.c_dims + self.s_toy3, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)
        #self.modules['mod6'] = LearningModule("mod6", self.m_arm, self.c_dims + self.s_sound, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)
        
        #self.modules['mod10'] = LearningModule("mod10", self.m_diva, self.c_dims + self.s_toy1, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)
        #self.modules['mod11'] = LearningModule("mod11", self.m_diva, self.c_dims + self.s_toy2, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)
        #self.modules['mod12'] = LearningModule("mod12", self.m_diva, self.c_dims + self.s_toy3, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)
        self.modules['mod13'] = LearningModule("mod13", self.m_diva, self.c_dims + self.s_sound, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)
        
        
        #self.modules['mod7'] = LearningModule("mod7", self.m_arm, self.c_dims + self.s_caregiver, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)        
        #self.modules['mod8'] = LearningModule("mod8", self.m_diva, self.c_dims + self.s_hand, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)
        #self.modules['mod9'] = LearningModule("mod9", self.m_diva, self.c_dims + self.s_tool, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)        
        #self.modules['mod14'] = LearningModule("mod14", self.m_diva, self.c_dims + self.s_caregiver, self.conf, context_mode=dict(mode='mcs', context_n_dims=10, context_sensory_bounds=[[-1.]*10,[1.]*10]), explo_noise=self.explo_noise)  
         

        for mid in self.modules.keys():
            self.progresses_evolution[mid] = []
            self.interests_evolution[mid] = []
            
        self.count_arm = 0
        self.count_diva = 0
        
        self.mids = ["mod"+ str(i) for i in range(1, 15) if "mod"+ str(i) in self.modules.keys()]

    
    def mid2motor_space(self, mid):
        if mid in ["mod"+ str(i) for i in range(1, 8)]:
            return "arm"
        else:
            return "diva"
    
    def save(self):
        sm_data = {}
        im_data = {}
        for mid in self.modules.keys():
            sm_data[mid] = self.modules[mid].sensorimotor_model.save()
            im_data[mid] = self.modules[mid].interest_model.save()            
        return {"sm_data":sm_data,
                "im_data":im_data,
                "chosen_modules":self.chosen_modules,
                "progresses_evolution":self.progresses_evolution,
                "interests_evolution":self.interests_evolution,
                "normalized_interests_evolution":self.get_normalized_interests_evolution()}

        
    def choose_babbling_module(self):
        if self.model_babbling == "random":
            mode = "random"
        elif self.model_babbling == "active":
            mode = "prop"
        interests = {}
        for mid in self.modules.keys():
            interests[mid] = self.modules[mid].interest()
        
        if mode == 'random':
            mid = np.random.choice(interests.keys())
        elif mode == 'greedy':
            if np.random.random() < self.choice_eps:
                mid = np.random.choice(interests.keys())
            else:
                mid = max(interests, key=interests.get)
        elif mode == 'softmax':
            temperature = self.choice_eps
            w = interests.values()
            mid = self.modules.keys()[softmax_choice(w, temperature)]
        
        elif mode == 'prop':
            w = interests.values()
            mid = self.modules.keys()[prop_choice(w, eps=self.choice_eps)]
        
        self.chosen_modules.append(mid)
        return mid
              
        
    def eval_mode(self): 
        self.sm_modes = {}
        for mod in self.modules.values():
            self.sm_modes[mod.mid] = mod.sensorimotor_model.mode
            mod.sensorimotor_model.mode = 'exploit'
                
    def learning_mode(self): 
        for mod in self.modules.values():
            mod.sensorimotor_model.mode = self.sm_modes[mod.mid]
                
    def motor_primitive(self, m): return m
    def sensory_primitive(self, s): return s
    def get_m(self, ms): return ms[self.conf.m_dims]
    def get_s(self, ms): return ms[self.conf.s_dims]
    
    def motor_babbling(self, arm=False, audio=False):
        self.m = rand_bounds(self.conf.m_bounds)[0]
        if arm:
            r = 1.
        elif audio:
            r = 0.
        else:
            r = np.random.random()
        if r < 0.5:
            self.m[:self.arm_n_dims] = 0.
        else:
            self.m[self.arm_n_dims:] = 0.
        return self.m
    
    def set_ms(self, m, s): return np.array(list(m) + list(s))
            
    def update_sensorimotor_models(self, ms):
        for mid in self.modules.keys():
            self.modules[mid].update_sm(self.modules[mid].get_m(ms), self.modules[mid].get_s(ms))
        
    def produce(self, context):
        if self.t < self.n_motor_babbling:
            self.mid_control = None
            self.chosen_modules.append("motor_babbling")
            return self.motor_babbling()
        else:
            mid = self.choose_babbling_module()
            self.mid_control = mid
            
            m = self.modules[mid].produce(context=context)
            if self.mid2motor_space(mid) == "arm":
                self.m = list(m) + [0.]*self.diva_n_dims
                self.count_arm += 1
            else:
                self.m = [0.]*self.arm_n_dims + list(m)
                self.count_diva += 1
            return self.m
    
    def perceive(self, s):
        s = self.sensory_primitive(s)
        ms = self.set_ms(self.m, s)
        self.update_sensorimotor_models(ms)
        if self.mid_control is not None:
            self.modules[self.mid_control].update_im(self.modules[self.mid_control].get_m(ms), self.modules[self.mid_control].get_s(ms))
        self.t = self.t + 1
        
        for mid in self.modules.keys():
            self.progresses_evolution[mid].append(self.modules[mid].progress())
            self.interests_evolution[mid].append(self.modules[mid].interest())            

    def get_normalized_interests_evolution(self):
        data = np.transpose(np.array([self.interests_evolution[mid] for mid in self.mids]))
        data_sum = data.sum(axis=1)
        data_sum[data_sum==0.] = 1.
        return data / data_sum.reshape(data.shape[0],1)
    
    def get_normalized_interests(self):
        interests = {}
        for mid in self.modules.keys():
            interests[mid] = self.modules[mid].interest()
            
        s = sum(interests.values())
        if s > 0:
            for mid in self.modules.keys():
                interests[mid] = interests[mid] / s
        return interests
        
    def print_stats(self):
        print "\n----------------\nAgent Statistics\n----------------\n"
        print "#Iterations:", self.t
        print
        for mid in self.mids:
            print "# Chosen module", mid, ":", self.chosen_modules.count(mid)
        print
        for mid in self.mids:
            print "Competence progress of", mid, ": " if mid in ["mod10", "mod11", "mod12", "mod13", "mod14"] else " : ", self.modules[mid].interest_model.current_competence_progress
        print
        for mid in self.mids:
            print "Prediction progress of", mid, ": " if mid in ["mod10", "mod11", "mod12", "mod13", "mod14"] else " : ", self.modules[mid].interest_model.current_prediction_progress
        print "#Arm trials", self.count_arm
        print "#Vocal trials", self.count_diva
        print
        
        
        
