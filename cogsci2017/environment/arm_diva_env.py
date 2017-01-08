import time
import numpy as np
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
        
        
        # DIVA CONFIG
        
        self.sound_o = list(np.log2([500, 900]))
        self.sound_y = list(np.log2([300, 1700]))
        self.sound_u = list(np.log2([300, 800]))
        self.sound_e = list(np.log2([600, 1700]))
        
        diva_cfg = dict(
                        m_mins = np.array([-1, -1, -1, -1, -1, -1, -1]),
                        m_maxs = np.array([1, 1, 1, 1, 1, 1, 1]),
                        s_mins = np.array([ 7.,  9.]),
                        s_maxs = np.array([ 10. ,  12.]),
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
        
        self.parent_gives_obj_factor = 0.99
        self.tool_length = 0.5
        self.handle_tol = 0.1
        self.handle_tol_sq = self.handle_tol * self.handle_tol
        self.handle_noise = 0.
        self.object_tol_hand = 0.1
        self.object_tol_hand_sq = self.object_tol_hand * self.object_tol_hand
        self.object_tol_tool = 0.1
        self.object_tol_tool_sq = self.object_tol_tool * self.object_tol_tool
        
        self.formant_tol = 0.1
        self.formants = None
        self.vowel = None
        self.diva_traj = None
        
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
        self.count_diva_o = 0
        self.count_diva_u = 0
        self.count_diva_e = 0
        self.count_parent_give_object = 0
        self.time_arm = 0.
        self.time_diva = 0. 
        self.time_arm_per_it = 0.
        self.time_diva_per_it = 0. 
        
        self.t = 0
          

    def reset(self):
        self.current_tool[3] = 0.
        self.current_toy1[2] = 0.
        self.current_toy2[2] = 0.
        self.current_toy3[2] = 0.
        
        self.hand = []
        self.tool = []
        self.toy1 = []
        self.toy2 = []
        self.toy3 = []
        self.sound = []
        self.caregiver = []
        
        self.current_context = self.get_current_context()  
        
        self.logs_tool = []
        self.logs_toy1 = []
        self.logs_toy2 = []
        self.logs_toy3 = []
        self.logs_caregiver = []
        
    def get_current_context(self):
        return self.current_tool[:2] + self.current_toy1[:2] + self.current_toy2[:2] + self.current_toy3[:2] + self.current_caregiver
    
    def compute_tool(self):
        a = np.pi * self.current_tool[2]
        self.tool_end_pos = [self.current_tool[0] + np.cos(a) * self.tool_length, 
                        self.current_tool[1] + np.sin(a) * self.tool_length]
        
    def compute_motor_command(self, m_ag):
        return bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)
    
    def motor_babbling(self):
        m = rand_bounds(self.conf.m_bounds)[0]
        if np.random.random() < 0.5:
            m[:21] = 0.
        else:
            m[21:] = 0.
        return m
    
    def is_hand_free(self):
        return self.current_tool[3] == 0 and (not self.current_toy1[2] == 1) and (not self.current_toy2[2] == 1) and (not self.current_toy3[2] == 1)
    
    def is_tool_free(self):
        return (not self.current_toy1[2] == 2) and (not self.current_toy2[2] == 2) and (not self.current_toy3[2] == 2) 

    def give_label(self, toy, toy_log):
        for i in range(self.timesteps):
            onset = i
            if toy_log[i][0][2] > 0:
                break
        
        if toy == "toy1":
            return [[0.,0.]] * onset + [self.sound_o] * (50 - onset)
        elif toy == "toy2":
            return [[0.,0.]] * onset + [self.sound_y] * (50 - onset)
        elif toy == "toy3":
            return [[0.,0.]] * onset + [self.sound_u] * (50 - onset)
        else:
            raise NotImplementedError
        
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
            #print "diva traj", diva_traj
            
            if (self.sound_o[0] - self.formant_tol < diva_traj[-1][0] < self.sound_o[0] + self.formant_tol) and (self.sound_o[1] - self.formant_tol < diva_traj[-1][1] < self.sound_o[1] + self.formant_tol):
                vowel = "o"
            elif (self.sound_u[0] - self.formant_tol < diva_traj[-1][0] < self.sound_u[0] + self.formant_tol) and (self.sound_u[1] - self.formant_tol < diva_traj[-1][1] < self.sound_u[1] + self.formant_tol):
                vowel = "u"
            elif (self.sound_e[0] - self.formant_tol < diva_traj[-1][0] < self.sound_e[0] + self.formant_tol) and (self.sound_e[1] - self.formant_tol < diva_traj[-1][1] < self.sound_e[1] + self.formant_tol):
                vowel = "e"
            else:
                vowel = None
            #print "diva vowel:", vowel
        else:
            diva_traj = np.zeros((50,2))
            vowel = None
        self.vowel = vowel
        self.formants = diva_traj[-1]
        
        
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
                if self.vowel == "o":
                    self.current_toy1[0] = self.current_toy1[0] * self.parent_gives_obj_factor
                    self.current_toy1[1] = self.current_toy1[1] * self.parent_gives_obj_factor
                elif self.vowel == "u":
                    self.current_toy2[0] = self.current_toy2[0] * self.parent_gives_obj_factor
                    self.current_toy2[1] = self.current_toy2[1] * self.parent_gives_obj_factor
                elif self.vowel == "e":
                    self.current_toy3[0] = self.current_toy3[0] * self.parent_gives_obj_factor
                    self.current_toy3[1] = self.current_toy3[1] * self.parent_gives_obj_factor
                    
                
                self.logs_toy1.append([self.current_toy1])
                self.logs_toy2.append([self.current_toy2])
                self.logs_toy3.append([self.current_toy3])
                self.logs_caregiver.append([self.current_caregiver])
        
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
                label = self.give_label("toy1", self.logs_toy1)
            elif self.current_toy2[2] == 1:
                label = self.give_label("toy2", self.logs_toy2)
            elif self.current_toy3[2] == 1:
                label = self.give_label("toy3", self.logs_toy3)
            else:
                label = [[0.,0.]] * 50
            self.sound = [f for formants in label[[0, 12, 24, 37, 49]] for f in formants]
            #print "parent sound", label, sound
        else:
            self.sound = [f for formants in diva_traj[[0, 12, 24, 37, 49]] for f in formants]
        
            
        
        # Sort dims            
        self.hand = self.hand[0::2] + self.hand[1::2]
        self.tool = self.tool[0::2] + self.tool[1::2]
        self.toy1 = self.toy1[0::2] + self.toy1[1::2]
        self.toy2 = self.toy2[0::2] + self.toy2[1::2]
        self.toy3 = self.toy3[0::2] + self.toy3[1::2]
        self.caregiver = self.caregiver[0::2] + self.caregiver[1::2]
        self.sound = self.sound[0::2] + self.sound[1::2]
        

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
        if self.vowel == "o":
            self.count_diva_o += 1
        elif self.vowel == "u":
            self.count_diva_u += 1
        elif self.vowel == "e":
            self.count_diva_e += 1
        self.count_parent_give_object = self.count_diva_o + self.count_diva_u + self.count_diva_e
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
        
        s = self.current_context + self.hand + self.tool + self.toy1 + self.toy2 + self.toy3 + self.sound + self.caregiver
        
        #print "len(s)", len(s)
        
        return s
    
    def print_stats(self):
        print"\n----------------------\nEnvironment Statistics\n----------------------\n"
        print "#Iterations:", self.t
        print "#Arm trials:", self.count_arm
        print "#Vocal trials:", self.count_diva
        print "#Tool actions:", self.count_tool
        print "#Toy1 was reached by tool:", self.count_toy1_by_tool
        print "#Toy2 was reached by tool:", self.count_toy2_by_tool
        print "#Toy3 was reached by tool:", self.count_toy3_by_tool
        print "#Toy1 was reached by hand:", self.count_toy1_by_hand
        print "#Toy2 was reached by hand:", self.count_toy2_by_hand
        print "#Toy3 was reached by hand:", self.count_toy3_by_hand
        print "#Parent gave vocal labels:", self.count_parent_give_label
        print "#Vowel /o/:", self.count_diva_o
        print "#Vowel /u/:", self.count_diva_u
        print "#Vowel /e/:", self.count_diva_e
        print "#Parent gave object:", self.count_parent_give_object
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
    
    