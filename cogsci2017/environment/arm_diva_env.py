import time
import numpy as np
import random
import matplotlib.pyplot as plt

from diva import DivaEnvironment
from arm_env import ArmEnvironment
from explauto.utils import bounds_min_max
from explauto.environment.environment import Environment
from explauto.utils.utils import rand_bounds

import brewer2mpl
bmap = brewer2mpl.get_map('Dark2', 'qualitative', 6)
colors = bmap.mpl_colors
  
colors_config = {
                 "stick":colors[1],
                 "gripper":colors[1],
                 "magnetic":colors[2],
                 "scratch":colors[4],
                 }
    
        

class CogSci2017Environment(Environment):
    def __init__(self, gui=False, audio=False):
        
        self.t = 0
        
        # ARM CONFIG
        
        arm_cfg = dict(
                        m_mins=[-1.] * 3 * 7, 
                        m_maxs=[1.] * 3 * 7, 
                        s_mins=[-1.] * 3 * 50,
                        s_maxs=[1.] * 3 * 50,
                        lengths=[0.5, 0.3, 0.2], 
                        angle_shift=0.5,
                        rest_state=[0., 0., 0.],
                        n_dmps=3, 
                        n_bfs=6, 
                        timesteps=50,
                        gui=gui)
        
        
        # SOUND CONFIG
        
        self.v_o = list(np.log2([500, 900]))
        self.v_y = list(np.log2([300, 1700]))
        self.v_u = list(np.log2([300, 800]))
        self.v_e = list(np.log2([400, 2200]))
        self.v_i = list(np.log2([300, 2300]))
        
        
        self.vowels = dict(o=self.v_o, y=self.v_y, u=self.v_u, e=self.v_e, i=self.v_i)
        
        self.human_sounds = ['oey', 'uye', 'iuo', 'eyu', 'eou', 'yeo']
        random.shuffle(self.human_sounds)
        print "human sounds", self.human_sounds
        
        
        def compute_s_sound(sound):
            s1 = self.vowels[sound[0]]
            s2 = [(self.vowels[sound[0]][0] + self.vowels[sound[1]][0]) / 2., (self.vowels[sound[0]][1] + self.vowels[sound[1]][1]) / 2.]
            s3 = self.vowels[sound[1]]
            s4 = [(self.vowels[sound[1]][0] + self.vowels[sound[2]][0]) / 2., (self.vowels[sound[1]][1] + self.vowels[sound[2]][1]) / 2.]
            s5 = self.vowels[sound[2]]
            rdm = 0.0 * (2.*np.random.random((1,10))[0] - 1.)
            return list(rdm + np.array([f[0] for f in [s1, s2, s3, s4, s5]] + [f[1] for f in [s1, s2, s3, s4, s5]]))
        
        
        self.human_sounds_traj = dict()
        self.human_sounds_traj_std = dict()
        self.best_vocal_errors = {}
        self.best_vocal_errors_evolution = []
        for hs in self.human_sounds:
            self.best_vocal_errors[hs] = 10.
            self.human_sounds_traj[hs] = compute_s_sound(hs)
            self.human_sounds_traj_std[hs] = [d - 8.5 for d in self.human_sounds_traj[hs][:5]] + [d - 10.25 for d in self.human_sounds_traj[hs][5:]]    
    
            
        self.sound_tol = 0.4
    

        # DIVA CONFIG
        
        diva_cfg = dict(
                        m_mins = np.array([-1, -1, -1, -1, -1, -1, -1]),
                        m_maxs = np.array([1, 1, 1, 1, 1, 1, 1]),
                        s_mins = np.array([ 7.5,  9.25]),
                        s_maxs = np.array([ 9.5 ,  11.25]),
                        m_used = range(7),
                        s_used = range(1, 3),
                        rest_position_diva = list([0]*7),
                        audio = audio,
                        diva_use_initial = True,
                        diva_use_goal = True,
                        used_diva = list([True]*7),
                        n_dmps_diva = 7,
                        n_bfs_diva = 2,
                        move_steps = 50,
                        )
        
        
        
        self.arm = ArmEnvironment(**arm_cfg)
        self.diva = DivaEnvironment(**diva_cfg)
        
        self.timesteps = 50
        
        
        # OBJECTS CONFIG
        
        self.caregiver_gives_obj_factor = 0.01
        self.tool_length = 0.5
        self.handle_tol = 0.2
        self.handle_tol_sq = self.handle_tol * self.handle_tol
        self.handle_noise = 0.
        self.object_tol_hand = 0.1
        self.object_tol_hand_sq = self.object_tol_hand * self.object_tol_hand
        self.object_tol_tool = 0.1
        self.object_tol_tool_sq = self.object_tol_tool * self.object_tol_tool
        
        self.diva_traj = None
        self.produced_sound = None
        
        Environment.__init__(self, 
                             m_mins= [-1.] * (21+28),
                             m_maxs= [1.] * (21+28),
                             s_mins= [-1.] * 80,
                             s_maxs= [1.] * 80)
        
        
        self.current_tool = [-0.5, 0., 0.5, 0.]
        self.current_toy1 = [0.5, 0.5, 0.]
        self.current_toy2 = [0.7, 0.7, 0.]
        self.current_toy3 = [0.9, 0.9, 0.]
        self.current_caregiver = [0., 1.7]
        self.reset()
        self.compute_tool()
        
        if self.arm.gui:
            self.init_plot()
            
        

        self.count_diva = 0
        self.count_arm = 0
        self.count_tool = 0
        self.count_toy1_by_tool = 0
        self.count_toy2_by_tool = 0
        self.count_toy3_by_tool = 0
        self.count_toy1_by_hand = 0
        self.count_toy2_by_hand = 0
        self.count_toy3_by_hand = 0
        self.count_parent_give_label = 0
        self.count_parent_give_object = 0
        self.count_produced_sounds = {}
        for hs in self.human_sounds:
            self.count_produced_sounds[hs] = 0
        
        self.time_arm = 0.
        self.time_diva = 0. 
        self.time_arm_per_it = 0.
        self.time_diva_per_it = 0. 
        
          
          
    def save(self):
        return dict(t=self.t,
                    human_sounds=self.human_sounds,
                    best_vocal_errors=self.best_vocal_errors,
                    best_vocal_errors_evolution=self.best_vocal_errors_evolution,
                    count_diva=self.count_diva,
                    count_arm=self.count_arm,
                    count_tool=self.count_tool,
                    count_toy1_by_tool=self.count_toy1_by_tool,
                    count_toy2_by_tool=self.count_toy2_by_tool,
                    count_toy3_by_tool=self.count_toy3_by_tool,
                    count_toy1_by_hand=self.count_toy1_by_hand,
                    count_toy2_by_hand=self.count_toy2_by_hand,
                    count_toy3_by_hand=self.count_toy3_by_hand,
                    count_parent_give_label=self.count_parent_give_label,
                    count_parent_give_object=self.count_parent_give_object,
                    count_produced_sounds=self.count_produced_sounds,
                    )

    def reset(self):
        
        if self.t % 20 == 0: 
            self.reset_toys()
        
        self.current_tool[3] = 0.
        self.current_toy1[2] = 0.
        self.current_toy2[2] = 0.
        self.current_toy3[2] = 0.
        self.reset_caregiver()
        self.current_context = self.get_current_context()  
        
        self.hand = []
        self.tool = []
        self.toy1 = []
        self.toy2 = []
        self.toy3 = []
        self.caregiver = []
        
        self.purge_logs()
        
    def purge_logs(self):
        self.logs_tool = []
        self.logs_toy1 = []
        self.logs_toy2 = []
        self.logs_toy3 = []
        self.logs_caregiver = []
        
    def reset_toys(self, region=0):
        self.current_toy1[:2] = self.reset_rand2d(region=region)
        self.current_toy2[:2] = self.reset_rand2d(region=region)
        self.current_toy3[:2] = self.reset_rand2d(region=region)
        
    def reset_caregiver(self):
        self.current_caregiver = self.reset_rand2d()
                
    def reset_rand2d(self, region=None):
        if region == 0:
            rdm = np.random.random()
            if rdm < 1. / 3.:
                return self.reset_rand2d(region=1)
            elif rdm < 2. / 3.:
                return self.reset_rand2d(region=2)
            else:
                return self.reset_rand2d(region=3)
        elif region == 1:
            alpha = 2. * np.pi * np.random.random()
            r = np.random.random()
            return [r * np.cos(alpha), r * np.sin(alpha)]
        elif region == 2:
            alpha = 2. * np.pi * np.random.random()
            r = 1. + 0.5 * np.random.random()
            return [r * np.cos(alpha), r * np.sin(alpha)]  
        elif region == 3:
            alpha = 2. * np.pi * np.random.random()
            r = 1.5 + 0.5 * np.random.random()
            return [r * np.cos(alpha), r * np.sin(alpha)]            
        elif region is None:
            return [4. * np.random.random() - 2., 4. * np.random.random() - 2.]
        
        
    def get_current_context(self):
        return [d / 2. for d in self.current_tool[:2] + self.current_toy1[:2] + self.current_toy2[:2] + self.current_toy3[:2] + self.current_caregiver]
    
    def compute_tool(self):
        a = np.pi * self.current_tool[2]
        self.tool_end_pos = [self.current_tool[0] + np.cos(a) * self.tool_length, 
                        self.current_tool[1] + np.sin(a) * self.tool_length]
        
    def compute_motor_command(self, m_ag):
        return bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)
    
    def motor_babbling(self, arm=False, audio=False):
        m = rand_bounds(self.conf.m_bounds)[0]
        if arm:
            r = 1.
        elif audio:
            r = 0.
        else:
            r = np.random.random()
        if r < 0.5:
            m[:21] = 0.
        else:
            m[21:] = 0.
        return m
    
    def is_hand_free(self):
        return self.current_tool[3] == 0 and (not self.current_toy1[2] == 1) and (not self.current_toy2[2] == 1) and (not self.current_toy3[2] == 1)
    
    def is_tool_free(self):
        return (not self.current_toy1[2] == 2) and (not self.current_toy2[2] == 2) and (not self.current_toy3[2] == 2) 

    def give_label(self, toy):
        
        if toy == "toy1":
            #print "Caregiver says", self.human_sounds[0] 
            return self.human_sounds_traj[self.human_sounds[0]]
        elif toy == "toy2":
            #print "Caregiver says", self.human_sounds[1]
            return self.human_sounds_traj[self.human_sounds[1]]
        elif toy == "toy3":
            #print "Caregiver says", self.human_sounds[2]
            return self.human_sounds_traj[self.human_sounds[2]]
        elif toy == "random":
            sound_id = np.random.choice([3, 4, 5])
            #print "Caregiver says", self.human_sounds[sound_id]
            return self.human_sounds_traj[self.human_sounds[sound_id]]
        else:
            raise NotImplementedError
        
    def analysis_sound(self, diva_traj):
        #return self.human_sounds[2]
        for hs in self.human_sounds:          
            error = np.linalg.norm(np.array(self.human_sounds_traj[hs]) - np.array([f[0] for f in diva_traj[[0, 12, 24, 37, 49]]] + [f[1] for f in diva_traj[[0, 12, 24, 37, 49]]]))
            if error < self.best_vocal_errors[hs]:
                self.best_vocal_errors[hs] = error
            if error < self.sound_tol:
                print "***********Agent says", hs
                return hs
        return None
    
    def caregiver_moves_obj(self, caregiver_pos, current_toy):
        middle = [caregiver_pos[0]/2, caregiver_pos[1]/2]
        return [current_toy[0] + self.caregiver_gives_obj_factor * (middle[0] - current_toy[0]), current_toy[1] + self.caregiver_gives_obj_factor * (middle[1] - current_toy[1]), current_toy[2]]
        
        
    def compute_sensori_effect(self, m):
        t = time.time()
        
        m_arm = m[:21]
        m_diva = m[21:]
        
        assert np.linalg.norm(m_arm) * np.linalg.norm(m_diva) == 0.
        
        if np.linalg.norm(m_arm) > 0.:
            cmd = "arm"
        else:
            cmd = "diva"
        
        arm_traj = self.arm.update(m_arm)
        #print "arm traj", arm_traj
        
        if cmd == "diva":
            diva_traj = self.diva.update(m_diva)
            self.diva_traj = diva_traj
            self.produced_sound = self.analysis_sound(self.diva_traj)
            if self.produced_sound is not None:
                self.count_produced_sounds[self.produced_sound] += 1
                if self.produced_sound in self.human_sounds[:3]:                    
                    self.count_parent_give_object += 1 
        else:
            diva_traj = np.zeros((50,2))
            self.produced_sound = None
        
        
        for i in range(self.timesteps):
            
            # Arm
            arm_x, arm_y, arm_angle = arm_traj[i]
            
            # Tool
            if not self.current_tool[3]:
                if self.is_hand_free() and ((arm_x - self.current_tool[0]) ** 2. + (arm_y - self.current_tool[1]) ** 2. < self.handle_tol_sq):
                    self.current_tool[0] = arm_x
                    self.current_tool[1] = arm_y
                    self.current_tool[2] = np.mod(arm_angle + self.handle_noise * np.random.randn() + 1, 2) - 1
                    self.compute_tool()
                    self.current_tool[3] = 1
            else:
                self.current_tool[0] = arm_x
                self.current_tool[1] = arm_y
                self.current_tool[2] = np.mod(arm_angle + self.handle_noise * np.random.randn() + 1, 2) - 1
                self.compute_tool()
            self.logs_tool.append([self.current_tool[:2], 
                          self.current_tool[2], 
                          self.tool_end_pos, 
                          self.current_tool[3]])
            
            
            if cmd == "arm":
                # Toy 1
                if self.current_toy1[2] == 1 or (self.is_hand_free() and ((arm_x - self.current_toy1[0]) ** 2 + (arm_y - self.current_toy1[1]) ** 2 < self.object_tol_hand_sq)):
                    self.current_toy1[0] = arm_x
                    self.current_toy1[1] = arm_y
                    self.current_toy1[2] = 1
                if self.current_toy1[2] == 2 or ((not self.current_toy1[2] == 1) and self.is_tool_free() and ((self.tool_end_pos[0] - self.current_toy1[0]) ** 2 + (self.tool_end_pos[1] - self.current_toy1[1]) ** 2 < self.object_tol_tool_sq)):
                    self.current_toy1[0] = self.tool_end_pos[0]
                    self.current_toy1[1] = self.tool_end_pos[1]
                    self.current_toy1[2] = 2
                self.logs_toy1.append([self.current_toy1])
                
                # Toy 2
                if self.current_toy2[2] == 1 or (self.is_hand_free() and ((arm_x - self.current_toy2[0]) ** 2 + (arm_y - self.current_toy2[1]) ** 2 < self.object_tol_hand_sq)):
                    self.current_toy2[0] = arm_x
                    self.current_toy2[1] = arm_y
                    self.current_toy2[2] = 1
                if self.current_toy2[2] == 2 or ((not self.current_toy2[2] == 1) and self.is_tool_free() and ((self.tool_end_pos[0] - self.current_toy2[0]) ** 2 + (self.tool_end_pos[1] - self.current_toy2[1]) ** 2 < self.object_tol_tool_sq)):
                    self.current_toy2[0] = self.tool_end_pos[0]
                    self.current_toy2[1] = self.tool_end_pos[1]
                    self.current_toy2[2] = 2
                self.logs_toy2.append([self.current_toy2])
                
                # Toy 3
                if self.current_toy3[2] == 1 or (self.is_hand_free() and ((arm_x - self.current_toy3[0]) ** 2 + (arm_y - self.current_toy3[1]) ** 2 < self.object_tol_hand_sq)):
                    self.current_toy3[0] = arm_x
                    self.current_toy3[1] = arm_y
                    self.current_toy3[2] = 1
                if self.current_toy3[2] == 2 or ((not self.current_toy3[2] == 1) and self.is_tool_free() and ((self.tool_end_pos[0] - self.current_toy3[0]) ** 2 + (self.tool_end_pos[1] - self.current_toy3[1]) ** 2 < self.object_tol_tool_sq)):
                    self.current_toy3[0] = self.tool_end_pos[0]
                    self.current_toy3[1] = self.tool_end_pos[1]
                    self.current_toy3[2] = 2
                self.logs_toy3.append([self.current_toy3])
                
                self.logs_caregiver.append([self.current_caregiver])
                
        
            else:
                # parent gives object if label is produced
                if self.produced_sound == self.human_sounds[0]:
                    self.current_toy1 =  self.caregiver_moves_obj(self.current_caregiver, self.current_toy1)
                elif self.produced_sound == self.human_sounds[1]:
                    self.current_toy2 =  self.caregiver_moves_obj(self.current_caregiver, self.current_toy2)
                elif self.produced_sound == self.human_sounds[2]:
                    self.current_toy3 =  self.caregiver_moves_obj(self.current_caregiver, self.current_toy3)
                    
                
                self.logs_toy1.append([self.current_toy1])
                self.logs_toy2.append([self.current_toy2])
                self.logs_toy3.append([self.current_toy3])
                self.logs_caregiver.append([self.current_caregiver])
        
#             if i in [0, 10, 20, 30, 40, 49]:
#                 self.hand = self.hand + [arm_x, arm_y]
#                 self.tool = self.tool + self.current_tool[:2]
#                 self.toy1 = self.toy1 + self.current_toy1[:2]
#                 self.toy2 = self.toy2 + self.current_toy2[:2]
#                 self.toy3 = self.toy3 + self.current_toy3[:2]
                
            if i in [0, 12, 24, 37, 49]:
                self.hand = self.hand + [arm_x, arm_y]
                self.tool = self.tool + self.current_tool[:2]
                self.toy1 = self.toy1 + self.current_toy1[:2]
                self.toy2 = self.toy2 + self.current_toy2[:2]
                self.toy3 = self.toy3 + self.current_toy3[:2]
                self.caregiver = self.caregiver + self.current_caregiver
        
            if self.arm.gui:
                if i % 5 == 0:
                    self.plot_step(self.ax, i)
                    
                
        if cmd == "arm":
            # parent gives label if object is touched by hand 
            if self.current_toy1[2] == 1:
                label = self.give_label("toy1")
            elif self.current_toy2[2] == 1:
                label = self.give_label("toy2")
            elif self.current_toy3[2] == 1:
                label = self.give_label("toy3")
            else:
                label = self.give_label("random")
            self.sound = label
            #print "parent sound", label, self.sound
        else:
            self.sound = [f for formants in diva_traj[[0, 12, 24, 37, 49]] for f in formants]
            self.sound = self.sound[0::2] + self.sound[1::2]
            
        
        # Sort dims            
        self.hand = self.hand[0::2] + self.hand[1::2]
        self.tool = self.tool[0::2] + self.tool[1::2]
        self.toy1 = self.toy1[0::2] + self.toy1[1::2]
        self.toy2 = self.toy2[0::2] + self.toy2[1::2]
        self.toy3 = self.toy3[0::2] + self.toy3[1::2]
        self.caregiver = self.caregiver[0::2] + self.caregiver[1::2]
        #self.sound = self.sound[0::2] + self.sound[1::2]
        

        # Analysis
        if np.linalg.norm(m[21:]) > 0:
            self.count_diva += 1
        else:
            self.count_arm += 1
        if self.current_tool[3]:
            self.count_tool += 1
        if self.current_toy1[2] == 1:
            self.count_toy1_by_hand += 1
        elif self.current_tool[3] and self.current_toy1[2] == 2:
            self.count_toy1_by_tool += 1
        if self.current_toy2[2] == 1:
            self.count_toy2_by_hand += 1
        elif self.current_tool[3] and self.current_toy2[2] == 2:
            self.count_toy2_by_tool += 1
        if self.current_toy3[2] == 1:
            self.count_toy3_by_hand += 1
        elif self.current_tool[3] and self.current_toy3[2] == 2:
            self.count_toy3_by_tool += 1
        self.count_parent_give_label = self.count_toy1_by_hand + self.count_toy2_by_hand + self.count_toy3_by_hand

        if cmd == "arm":
            self.time_arm += time.time() - t
            self.time_arm_per_it = self.time_arm / self.count_arm
        else:
            self.time_diva += time.time() - t
            self.time_diva_per_it = self.time_diva / self.count_diva
            
        #print "previous context", len(self.current_context), self.current_context
        #print "s_hand", len(self.hand), self.hand
        #print "s_tool", len(self.tool), self.tool
        #print "s_toy1", len(self.toy1), self.toy1
        #print "s_toy2", len(self.toy2), self.toy2
        #print "s_toy3", len(self.toy3), self.toy3
        #print "s_sound", len(self.sound), self.sound
        #print "s_caregiver", len(self.caregiver), self.caregiver
        
        self.t += 1
        if self. t % 100 == 0:
            self.best_vocal_errors_evolution += [self.best_vocal_errors.copy()]
        if self.t % 1000 == 0:
            print "best_vocal_errors", [(hs,self.best_vocal_errors[hs]) for hs in self.human_sounds]
        
        
        context = self.current_context
        
        # MAP TO STD INTERVAL
        hand = [d/2 for d in self.hand]
        tool = [d/2 for d in self.tool]
        toy1 = [d/2 for d in self.toy1]
        toy2 = [d/2 for d in self.toy2]
        toy3 = [d/2 for d in self.toy3]
        sound = [d - 8.5 for d in self.sound[:5]] + [d - 10.25 for d in self.sound[5:]]
        caregiver = [d/2 for d in self.caregiver]
        
        # MAP to Delta
#         hand = [hand[1] - hand[0], hand[2] - hand[1], hand[3] - hand[2], hand[4] - hand[3], hand[5] - hand[4],
#                 hand[7] - hand[6], hand[8] - hand[7], hand[9] - hand[8], hand[10] - hand[9], hand[11] - hand[10]]
#         
#         tool = [tool[1] - tool[0], tool[2] - tool[1], tool[3] - tool[2], tool[4] - tool[3], tool[5] - tool[4],
#                 tool[7] - tool[6], tool[8] - tool[7], tool[9] - tool[8], tool[10] - tool[9], tool[11] - tool[10]]
#         
#         toy1 = [toy1[1] - toy1[0], toy1[2] - toy1[1], toy1[3] - toy1[2], toy1[4] - toy1[3], toy1[5] - toy1[4],
#                 toy1[7] - toy1[6], toy1[8] - toy1[7], toy1[9] - toy1[8], toy1[10] - toy1[9], toy1[11] - toy1[10]]
#         
#         toy2 = [toy2[1] - toy2[0], toy2[2] - toy2[1], toy2[3] - toy2[2], toy2[4] - toy2[3], toy2[5] - toy2[4],
#                 toy2[7] - toy2[6], toy2[8] - toy2[7], toy2[9] - toy2[8], toy2[10] - toy2[9], toy2[11] - toy2[10]]
#         
#         toy3 = [toy3[1] - toy3[0], toy3[2] - toy3[1], toy3[3] - toy3[2], toy3[4] - toy3[3], toy3[5] - toy3[4],
#                 toy3[7] - toy3[6], toy3[8] - toy3[7], toy3[9] - toy3[8], toy3[10] - toy3[9], toy3[11] - toy3[10]]
                
        
        
        s = context + hand + tool + toy1 + toy2 + toy3 + sound + caregiver
        #print "s_sound", sound
        return bounds_min_max(s, self.conf.s_mins, self.conf.s_maxs)
    
    
    def update(self, m_ag, reset=True, log=True):
        """ Computes sensorimotor values from motor orders.

        :param numpy.array m_ag: a motor command with shape (self.conf.m_ndims, ) or a set of n motor commands of shape (n, self.conf.m_ndims)

        :param bool log: emit the motor and sensory values for logging purpose (default: True).

        :returns: an array of shape (self.conf.ndims, ) or (n, self.conf.ndims) according to the shape of the m_ag parameter, containing the motor values (which can be different from m_ag, e.g. bounded according to self.conf.m_bounds) and the corresponding sensory values.

        .. note:: self.conf.ndims = self.conf.m_ndims + self.conf.s_ndims is the dimensionality of the sensorimotor space (dim of the motor space + dim of the sensory space).
        """

        if len(np.array(m_ag).shape) == 1:
            s = self.one_update(m_ag, log)
        else:
            s = []
            for m in m_ag:
                s.append(self.one_update(m, log))
            s = np.array(s)
        if reset:
            self.reset()
        return s
    

    def print_stats(self):
        print"\n----------------------\nEnvironment Statistics\n----------------------\n"
        print "# Iterations:", self.t
        print "# Arm trials:", self.count_arm
        print "# Vocal trials:", self.count_diva
        print "# Tool actions:", self.count_tool
        print "# Produced sounds:", self.count_produced_sounds
        print "# Toy1 was reached by tool:", self.count_toy1_by_tool
        print "# Toy2 was reached by tool:", self.count_toy2_by_tool
        print "# Toy3 was reached by tool:", self.count_toy3_by_tool
        print "# Toy1 was reached by hand:", self.count_toy1_by_hand
        print "# Toy2 was reached by hand:", self.count_toy2_by_hand
        print "# Toy3 was reached by hand:", self.count_toy3_by_hand
        print "# Parent gave vocal labels:", self.count_parent_give_label
        print "# Parent gave object:", self.count_parent_give_object
        print

    def init_plot(self):
        #plt.ion()
        self.ax = plt.subplot()
        plt.gcf().set_size_inches(6., 6., forward=True)
        plt.gca().set_aspect('equal')
        #plt.show(block=False)
        plt.axis([-2, 2,-2, 2])
        plt.draw()
    
    def plot_tool_step(self, ax, i, **kwargs_plot):
        handle_pos = self.logs_tool[i][0]
        end_pos = self.logs_tool[i][2]
        
        ax.plot([handle_pos[0], end_pos[0]], [handle_pos[1], end_pos[1]], '-', color=colors_config['stick'], lw=6, **kwargs_plot)
        ax.plot(handle_pos[0], handle_pos[1], 'o', color = colors_config['gripper'], ms=12, **kwargs_plot)
        ax.plot(end_pos[0], end_pos[1], 'o', color = colors_config['magnetic'], ms=12, **kwargs_plot)                    
    
    def plot_toy1_step(self, ax, i, **kwargs_plot):
        pos = self.logs_toy1[i][0]
        rectangle = plt.Rectangle((pos[0] - 0.1, pos[1] - 0.1), 0.2, 0.2, color = colors[3], **kwargs_plot)
        ax.add_patch(rectangle) 
    
    def plot_toy2_step(self, ax, i, **kwargs_plot):
        pos = self.logs_toy2[i][0]
        rectangle = plt.Rectangle((pos[0] - 0.1, pos[1] - 0.1), 0.2, 0.2, color = colors[4], **kwargs_plot)
        ax.add_patch(rectangle) 
    
    def plot_toy3_step(self, ax, i, **kwargs_plot):
        pos = self.logs_toy3[i][0]
        rectangle = plt.Rectangle((pos[0] - 0.1, pos[1] - 0.1), 0.2, 0.2, color = colors[5], **kwargs_plot)
        ax.add_patch(rectangle) 
        
    def plot_caregiver_step(self, ax, i, **kwargs_plot):
        pos = self.logs_caregiver[i][0]
        rectangle = plt.Rectangle((pos[0] - 0.1, pos[1] - 0.1), 0.2, 0.2, color = "black", **kwargs_plot)
        ax.add_patch(rectangle) 
        
    def plot_step(self, ax, i, **kwargs_plot):
        #t0 = time.time()
        plt.pause(0.0001)
        #print "t1", time.time() - t0
        plt.cla()
        #print "t2", time.time() - t0
        self.arm.plot_step(ax, i, **kwargs_plot)
        self.plot_tool_step(ax, i, **kwargs_plot)
        self.plot_toy1_step(ax, i, **kwargs_plot)
        self.plot_toy2_step(ax, i, **kwargs_plot)
        self.plot_toy3_step(ax, i, **kwargs_plot)
        self.plot_caregiver_step(ax, i, **kwargs_plot)
        #print "t3", time.time() - t0
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.xlabel("")
        plt.ylabel("")
        plt.draw()
        plt.show(block=False)
    
    