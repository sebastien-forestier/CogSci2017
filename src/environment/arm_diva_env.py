import time
import numpy as np
import matplotlib.pyplot as plt

from diva import DivaEnvironment
from arm_env import ArmEnvironment
from explauto.utils import bounds_min_max
from explauto.environment.environment import Environment
from explauto.utils.utils import rand_bounds

    
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
        
        self.tool_length = 0.5
        self.handle_tol = 0.3
        self.handle_tol_sq = self.handle_tol * self.handle_tol
        self.handle_noise = 0.
        self.tool_rest_state = [-0.5, 0., 0.5]
        self.object_tol_hand = 0.2
        self.object_tol_hand_sq = self.object_tol_hand * self.object_tol_hand
        self.object_tol_tool = 0.2
        self.object_tol_tool_sq = self.object_tol_tool * self.object_tol_tool
        
        self.formant_tol = 0.3
        self.formants = None
        self.vowel = None
        
        Environment.__init__(self, 
                             m_mins= [-1.] * (21+21),
                             m_maxs= [1.] * (21+21),
                             s_mins= [-1.] * 70,
                             s_maxs= [1.] * 70)
        
        
        self.reset()
        
        if self.arm.gui:
            self.init_plot()
            

    def reset(self):
        self.current_tool = [0., 0., 0.]
        self.current_toy1 = [0., 0., 0.]
        self.current_toy2 = [0., 0., 0.]
        self.current_toy3 = [0., 0., 0.]
        self.current_parent = [0., 0.]
        self.current_context = self.get_current_context()            
        self.hand = [0.] * 10
        self.tool = [0.] * 10
        self.toy1 = [0.] * 10
        self.toy2 = [0.] * 10
        self.toy3 = [0.] * 10
        self.sound = [0.] * 10
        
        self.handle_pos = np.array(self.tool_rest_state[0:2])
        self.tool_angle = self.tool_rest_state[2]
        self.compute_tool()
        self.logs = []
        
    def get_current_context(self):
        return self.current_tool[:2] + self.current_toy1[:2] + self.current_toy2[:2] + self.current_toy3[:2] + self.current_parent
    
    def compute_tool(self):
        a = np.pi * self.tool_angle
        self.tool_end_pos = [self.handle_pos[0] + np.cos(a) * self.tool_length, 
                        self.handle_pos[1] + np.sin(a) * self.tool_length]
        
    def compute_motor_command(self, m_ag):
        return bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)
    
    def motor_babbling(self):
        return rand_bounds(self.conf.m_bounds)[0]

    def compute_sensori_effect(self, m):
        arm_traj = self.arm.update(m[:21])
        #print "arm traj", arm_traj
        
        diva_traj = self.diva.update(m[21:])
        #print "diva traj", diva_traj
        
        vowel = None
        if (sound_o[0] - self.formant_tol < diva_traj[-1][0] < sound_o[0] + self.formant_tol) and (sound_o[1] - self.formant_tol < diva_traj[-1][1] < sound_o[1] + self.formant_tol):
            vowel =  "o"
        if (sound_u[0] - self.formant_tol < diva_traj[-1][0] < sound_u[0] + self.formant_tol) and (sound_u[1] - self.formant_tol < diva_traj[-1][1] < sound_u[1] + self.formant_tol):
            vowel =  "u"
        if (sound_e[0] - self.formant_tol < diva_traj[-1][0] < sound_e[0] + self.formant_tol) and (sound_e[1] - self.formant_tol < diva_traj[-1][1] < sound_e[1] + self.formant_tol):
            vowel =  "e"
        
        self.vowel = vowel
        self.formants = diva_traj[-1]
        #print "vowel:", vowel
        
        for i in range(self.timesteps):
            
            # Arm
            arm_x, arm_y, arm_angle = arm_traj[i]
            
            # Tool
            if not self.current_tool[2]:
                if (arm_x - self.handle_pos[0]) ** 2. + (arm_y - self.handle_pos[1]) ** 2. < self.handle_tol_sq:
                    self.handle_pos = [arm_x, arm_y]
                    self.angle = np.mod(arm_angle + self.handle_noise * np.random.randn() + 1, 2) - 1
                    self.compute_tool()
                    self.current_tool[2] = 1
            else:
                self.handle_pos = [arm_x, arm_y]
                self.angle = np.mod(arm_angle + self.handle_noise * np.random.randn() + 1, 2) - 1
                self.compute_tool()
            
            # Toy 1
            if self.current_toy1[2] == 1 or (self.current_toy1[2] == 0 and ((arm_x - self.current_toy1[0]) ** 2 + (arm_y - self.current_toy1[1]) ** 2 < self.object_tol_hand_sq)):
                self.current_toy1[0] = arm_x
                self.current_toy1[1] = arm_y
                self.current_toy1[2] = 1
            if self.current_toy1[2] == 2 or (self.current_toy1[2] == 0 and ((self.current_tool[0] - self.current_toy1[0]) ** 2 + (self.current_tool[1] - self.current_toy1[1]) ** 2 < self.object_tol_tool_sq)):
                self.current_toy1[0] = self.current_tool[0]
                self.current_toy1[1] = self.current_tool[1]
                self.current_toy1[2] = 2
            
            # Toy 2
            if self.current_toy2[2] == 1 or (self.current_toy2[2] == 0 and ((arm_x - self.current_toy2[0]) ** 2 + (arm_y - self.current_toy2[1]) ** 2 < self.object_tol_hand_sq)):
                self.current_toy2[0] = arm_x
                self.current_toy2[1] = arm_y
                self.current_toy2[2] = 1
            if self.current_toy2[2] == 2 or (self.current_toy2[2] == 0 and ((self.current_tool[0] - self.current_toy2[0]) ** 2 + (self.current_tool[1] - self.current_toy2[1]) ** 2 < self.object_tol_tool_sq)):
                self.current_toy2[0] = self.current_tool[0]
                self.current_toy2[1] = self.current_tool[1]
                self.current_toy2[2] = 2
            
            # Toy 3
            if self.current_toy3[2] == 1 or (self.current_toy3[2] == 0 and ((arm_x - self.current_toy3[0]) ** 2 + (arm_y - self.current_toy3[1]) ** 2 < self.object_tol_hand_sq)):
                self.current_toy3[0] = arm_x
                self.current_toy3[1] = arm_y
                self.current_toy3[2] = 1
            if self.current_toy3[2] == 2 or (self.current_toy3[2] == 0 and ((self.current_tool[0] - self.current_toy3[0]) ** 2 + (self.current_tool[1] - self.current_toy3[1]) ** 2 < self.object_tol_tool_sq)):
                self.current_toy3[0] = self.current_tool[0]
                self.current_toy3[1] = self.current_tool[1]
                self.current_toy3[2] = 2
        
        
            if self.arm.gui:
                self.plot_step(self.ax, i)
                
                
        self.current_tool[2] = 0
        self.current_toy1[2] = 0
        self.current_toy2[2] = 0
        self.current_toy3[2] = 0
        
        self.hand = [0.] * 10
        self.tool = [0.] * 10
        self.toy1 = [0.] * 10
        self.toy2 = [0.] * 10
        self.toy3 = [0.] * 10
        self.sound = [0.] * 10
        
        
        return self.current_context + self.hand + self.tool + self.toy1 + self.toy2 + self.toy3 + self.sound
    
    
    def init_plot(self):
        #plt.ion()
        self.ax = plt.subplot()
        plt.gcf().set_size_inches(6., 6., forward=True)
        plt.gca().set_aspect('equal')
        #plt.show(block=False)
        plt.axis([-2, 2,-2, 2])
        plt.draw()
    
    def plot_step(self, ax, i, **kwargs_plot):
        plt.pause(0.0001)
        plt.cla()
        self.arm.plot_step(ax, i, **kwargs_plot)
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
#             plt.xlim([-1.3, 1.3])
#             plt.ylim([-0.2, 1.6])
#             plt.xlim([-1.6, 1.6])
#             plt.ylim([-0.5, 1.6])
#             plt.gca().set_xticklabels([])
#             plt.gca().set_yticklabels([])
#             plt.gca().yaxis.set_major_locator(plt.NullLocator())
#             plt.gca().yaxis.set_major_locator(plt.NullLocator())
#             plt.xlabel("")
#             plt.ylabel("")
        plt.draw()
        plt.show(block=False)
    
    