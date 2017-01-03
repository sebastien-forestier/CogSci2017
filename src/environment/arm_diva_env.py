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
                gui=True)


# DIVA CONFIG

sound_o = list(np.log2([500, 900]))
sound_y = list(np.log2([300, 1700]))
sound_u = list(np.log2([300, 800]))
sound_e = list(np.log2([600, 1700]))

diva_cfg = dict(
                m_mins = np.array([-1, -1, -1, -1, -1, -1, -1]),
                m_maxs = np.array([1, 1, 1, 1, 1, 1, 1]),
                s_mins = np.array([ 7.,  9.]),
                s_maxs = np.array([ 10. ,  11.5]),
                m_used = range(7),
                s_used = range(1, 3),
                rest_position_diva = list([0]*7),
                audio = True,
                diva_use_initial = False,
                diva_use_goal = True,
                used_diva = list([True]*7),
                n_dmps_diva = 7,
                n_bfs_diva = 2,
                move_steps = 50,
                )
        

class CogSci2017Environment(Environment):
    def __init__(self):
        
        self.arm = ArmEnvironment(**arm_cfg)
        self.diva = DivaEnvironment(**diva_cfg)
        
        self.timesteps = 50
        
        self.parent_gives_obj_factor = 0.99
        self.tool_length = 0.5
        self.handle_tol = 0.5
        self.handle_tol_sq = self.handle_tol * self.handle_tol
        self.handle_noise = 0.
        self.object_tol_hand = 0.2
        self.object_tol_hand_sq = self.object_tol_hand * self.object_tol_hand
        self.object_tol_tool = 0.2
        self.object_tol_tool_sq = self.object_tol_tool * self.object_tol_tool
        
        self.formant_tol = 0.5
        self.formants = None
        self.vowel = None
        
        Environment.__init__(self, 
                             m_mins= [-1.] * (21+21),
                             m_maxs= [1.] * (21+21),
                             s_mins= [-1.] * 70,
                             s_maxs= [1.] * 70)
        
        
        self.current_tool = [-0.5, 0., 0.5, 0.]
        self.current_toy1 = [0.5, 0.5, 0.]
        self.current_toy2 = [0.7, 0.7, 0.]
        self.current_toy3 = [0.9, 0.9, 0.]
        self.current_parent = [1., 1.]
        self.reset()
        self.compute_tool()
        
        if self.arm.gui:
            self.init_plot()
            

    def reset(self):
        self.current_context = self.get_current_context()            
        self.hand = [0.] * 10
        self.tool = [0.] * 10
        self.toy1 = [0.] * 10
        self.toy2 = [0.] * 10
        self.toy3 = [0.] * 10
        self.sound = [0.] * 10
        
        self.current_tool[3] = 0.
        self.current_toy1[2] = 0.
        self.current_toy2[2] = 0.
        self.current_toy3[2] = 0.
        
        self.logs_tool = []
        self.logs_toy1 = []
        self.logs_toy2 = []
        self.logs_toy3 = []
        
    def get_current_context(self):
        return self.current_tool[:2] + self.current_toy1[:2] + self.current_toy2[:2] + self.current_toy3[:2] + self.current_parent
    
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
        print "random m", m
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
            return [[0.,0.]] * onset + [sound_o] * (50 - onset)
        elif toy == "toy2":
            return [[0.,0.]] * onset + [sound_y] * (50 - onset)
        elif toy == "toy3":
            return [[0.,0.]] * onset + [sound_u] * (50 - onset)
        else:
            raise NotImplementedError
        
    def compute_sensori_effect(self, m):
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
            #print "diva traj", diva_traj
            
            if (sound_o[0] - self.formant_tol < diva_traj[-1][0] < sound_o[0] + self.formant_tol) and (sound_o[1] - self.formant_tol < diva_traj[-1][1] < sound_o[1] + self.formant_tol):
                vowel = "o"
            elif (sound_u[0] - self.formant_tol < diva_traj[-1][0] < sound_u[0] + self.formant_tol) and (sound_u[1] - self.formant_tol < diva_traj[-1][1] < sound_u[1] + self.formant_tol):
                vowel = "u"
            elif (sound_e[0] - self.formant_tol < diva_traj[-1][0] < sound_e[0] + self.formant_tol) and (sound_e[1] - self.formant_tol < diva_traj[-1][1] < sound_e[1] + self.formant_tol):
                vowel = "e"
            else:
                vowel = None
            print "diva vowel:", vowel
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
        
            if self.arm.gui:
                self.plot_step(self.ax, i)
                
        # parent gives label if object is touched by hand 
        if self.current_toy1[2] == 1:
            label = self.give_label("toy1", self.logs_toy1)
        elif self.current_toy2[2] == 1:
            label = self.give_label("toy2", self.logs_toy2)
        elif self.current_toy3[2] == 1:
            label = self.give_label("toy3", self.logs_toy3)
        else:
            label = [[0.,0.]] * 50
        sound = [f for formants in label[0:-1:10] for f in formants]
        print "parent sound", label, sound
        
        
        self.hand = [0.] * 10
        self.tool = [0.] * 10
        self.toy1 = [0.] * 10
        self.toy2 = [0.] * 10
        self.toy3 = [0.] * 10
        self.sound = sound
                        
        return self.current_context + self.hand + self.tool + self.toy1 + self.toy2 + self.toy3 + self.sound
    
    
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
    
    