import time
import numpy as np
import matplotlib.pyplot as plt

from explauto.utils import bounds_min_max
from explauto.environment.environment import Environment
from explauto.environment.simple_arm.simple_arm import joint_positions
from grip_arm import GripArmEnvironment
from combined_env import CombinedEnvironment, HierarchicallyCombinedEnvironment
from dynamic_env import DynamicEnvironment
from tools import Stick, Box



class SceneObject(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 object_shape, object_tol1, object_tol2, rest_state):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.object_shape = object_shape
        self.object_tol1 = object_tol1
        self.object_tol1_sq = object_tol1 * object_tol1
        self.object_tol2 = object_tol2
        self.object_tol2_sq = object_tol2 * object_tol2
        self.rest_state = rest_state
        self.reset()
        
        
    def reset(self):
        self.move = 0
        self.pos = self.rest_state
        self.logs = []
        
    def compute_motor_command(self, m):
        #return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        return m

    def compute_sensori_effect(self, m):
        #print "SceneObject m", m
        if self.move == 1 or ((not self.move == 2) and (m[0] - self.pos[0]) ** 2 + (m[1] - self.pos[1]) ** 2 < self.object_tol1_sq):
            self.pos = m[0:2]
            self.move = 1
            #print "OBJECT PUSHED BY TOOL 1!"
        if self.move == 2 or ((not self.move == 1) and (m[2] - self.pos[0]) ** 2 + (m[3] - self.pos[1]) ** 2 < self.object_tol2_sq):
            self.pos = m[2:4]
            self.move = 2
            #print "OBJECT PUSHED BY TOOL 2!"
        self.logs.append([self.pos,
                          self.move])
        return list(self.pos)
    
    def update(self, m_traj, reset=False, log=True, batch=False):
        if reset:
            self.reset()
        s = []
        #print "SceneObject update m_traj", m_traj
        for m in m_traj:
            #print "SceneObject update m", m
            s.append(self.one_update(m, log))
        return s[-1]
    
    def plot(self, ax, i, **kwargs_plot):
        pos = self.logs[i][0]        
        if self.object_shape == "rectangle":
            rectangle = plt.Rectangle((pos[0] - 0.05, pos[1] - 0.05), 0.1, 0.1, fc='y')
            ax.add_patch(rectangle)
        elif self.object_shape == "circle":
            circle = plt.Circle((pos[0], pos[1]), 0.05, fc='y')
            ax.add_patch(circle)            
        else:
            raise NotImplementedError       
        

class CogSci2016Environment(DynamicEnvironment):
    def __init__(self, move_steps=50, max_params=None, perturbation=None, gui=False):

        def motor_perturbation(m):
            if perturbation == "BrokenMotor":
                m[2] = 0                
                return m
            elif perturbation == "ShiftedMotor":
                m[0] = m[0] - 0.3
                if m[0] < -1.:
                    m[0] = -1.
                return m
            else:
                return m
            
        gripArm_cfg = dict(m_mins=[-1, -1, -1, -1],  # joints pos + gripper state
                             m_maxs=[1, 1, 1, 1], 
                             s_mins=[-1, -1, -1, -1, -1], # hand pos + hand angle + gripper_change + gripper state
                             s_maxs=[1, 1, 1, 1, 1], 
                             lengths=[0.5, 0.3, 0.2], 
                             angle_shift=0.5,
                             rest_state=[0., 0., 0., 0.])
        
        stick1_cfg = dict(m_mins=[-1, -1, -1, -1, -1], 
                         m_maxs=[1, 1, 1, 1, 1], 
                         s_mins=[-2, -2],  # Tool pos
                         s_maxs=[2, 2],
                         length=0.6, 
                         handle_tol=0.2, 
                         handle_noise=0., 
                         rest_state=[-0.75, 0.25, 0.75],
                         perturbation=perturbation)
        
        stick2_cfg = dict(m_mins=[-1, -1, -1, -1, -1], 
                         m_maxs=[1, 1, 1, 1, 1], 
                         s_mins=[-2, -2], 
                         s_maxs=[2, 2],
                         length=0.3, 
                         handle_tol=0.2, 
                         handle_noise=0., 
                         rest_state=[0.75, 0.25, 0.25])
        
        sticks_cfg = dict(
                        s_mins = [-2, -2, -2, -2],
                        s_maxs = [2, 2, 2, 2],
                        envs_cls = [Stick, Stick],
                        envs_cfg = [stick1_cfg, stick2_cfg],
                        combined_s = lambda s:s  # from s:  Tool1 end pos + Tool2 end pos
                        )
        
        arm_stick_cfg = dict(m_mins=list([-1.] * 4), # 3DOF + gripper
                             m_maxs=list([1.] * 4),
                             s_mins=list([-2.] * 7),
                             s_maxs=list([2.] * 7),
                             top_env_cls=CombinedEnvironment, 
                             lower_env_cls=GripArmEnvironment, 
                             top_env_cfg=sticks_cfg, 
                             lower_env_cfg=gripArm_cfg, 
                             fun_m_lower= lambda m:[motor_perturbation(m_t[0:4]) for m_t in m],
                             fun_s_lower=lambda m,s:s+s,  # (hand pos + hand angle + gripper_change + gripper state) * 2 tools
                             fun_s_top=lambda m,s_lower,s:[s_l[0:2] + [s_l[4]] + s_ for s_l, s_ in zip(s_lower, s)]) # from s: Tool1 end pos + Tool2 end pos  from m: hand_pos + gripper state
        
        
        object_cfg = dict(m_mins = list([-1.] * 4), 
                          m_maxs = list([1.] * 4), 
                          s_mins = [-1., -1.], # new pos
                          s_maxs = [1., 1.],
                          object_shape = "circle", 
                          object_tol1 = 0.1, 
                          object_tol2 = 0.1, 
                          rest_state = [0., 1.2])
        
        arm_sticks_object_cfg = dict(
                                           m_mins=arm_stick_cfg['m_mins'],
                                           m_maxs=arm_stick_cfg['m_maxs'],
                                           s_mins=list([-2.] * (7 * move_steps + 2)),
                                           s_maxs=list([2.] * (7 * move_steps + 2)), # (hand pos + gripper state + tool1 end pos + tool2 end pos) traj + last object pos = 7 x samples + 2
                                           top_env_cls=SceneObject, 
                                           lower_env_cls=HierarchicallyCombinedEnvironment, 
                                           top_env_cfg=object_cfg, 
                                           lower_env_cfg=arm_stick_cfg, 
                                           fun_m_lower= lambda m:m,
                                           fun_s_lower=lambda m,s:s[3:],
                                           fun_s_top=lambda m,s_lower,s: [si[:7] for si in s_lower] + s)
        
        box1_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0.],
                     s_maxs = [1.],
                     box_m_mins = [-1.5, -0.1], 
                     box_m_maxs = [-1.3, 0.1]
                     )
        
        box2_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0.],
                     s_maxs = [1.],
                     box_m_mins = [-1.2, 1.], 
                     box_m_maxs = [-1., 1.2]
                     )
        
        box3_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0.],
                     s_maxs = [1.],
                     box_m_mins = [-0.1, 1.3], 
                     box_m_maxs = [0.1, 1.5]
                     )
        
        box4_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0.],
                     s_maxs = [1.],
                     box_m_mins = [1., 1.], 
                     box_m_maxs = [1.2, 1.2]
                     )        
        box5_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0.],
                     s_maxs = [1.],
                     box_m_mins = [1.3, -0.1], 
                     box_m_maxs = [1.5, 0.1]
                     )
        
        box6_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0.],
                     s_maxs = [1.],
                     box_m_mins = [-1., -0.1], 
                     box_m_maxs = [-0.8, 0.1]
                     )
        
        box7_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0.],
                     s_maxs = [1.],
                     box_m_mins = [-0.7, 0.5], 
                     box_m_maxs = [-0.5, 0.7]
                     )
        
        box8_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0.],
                     s_maxs = [1.],
                     box_m_mins = [-0.1, 0.8], 
                     box_m_maxs = [0.1, 1.]
                     )
        
        box9_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0.],
                     s_maxs = [1.],
                     box_m_mins = [0.5, 0.5], 
                     box_m_maxs = [0.7, 0.7]
                     )
        
        box10_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0.],
                     s_maxs = [1.],
                     box_m_mins = [0.8, -0.1], 
                     box_m_maxs = [1., 0.1]
                     )
        
         
        def combined_boxes(s):
            if s[0]:
                b = 1.
            elif s[2]:
                b = 2.
            elif s[4]:
                b = 3.
            elif s[6]:
                b = 4.
            elif s[8]:
                b = 5.
            elif s[10]:
                b = 6.
            elif s[12]:
                b = 7.
            elif s[14]:
                b = 8.
            elif s[16]:
                b = 9.
            elif s[18]:
                b = 10.
            else:
                b = 0.
            d = min([0.3, s[1], s[3], s[5], s[7], s[9], s[11], s[13], s[15], s[17], s[19]])
            return [b, d]
        
        boxes_cfg = dict(
                        s_mins = list([0., 0.]),
                        s_maxs = list([10., 2.]),
                        envs_cls = [Box, Box, Box, Box, Box, Box, Box, Box, Box, Box], 
                        envs_cfg = [box1_cfg, box2_cfg, box3_cfg, box4_cfg, box5_cfg, box6_cfg, box7_cfg, box8_cfg, box9_cfg, box10_cfg], 
                        combined_s = combined_boxes
                        )
        
        static_env_cfg = dict(m_mins=arm_stick_cfg['m_mins'],
                               m_maxs=arm_stick_cfg['m_maxs'],
                               s_mins=list([-2.] * (7 * move_steps + 2) + [0., 0.]), # 7x + 2 obj + 2 boxes
                               s_maxs=list([2.] * (7 * move_steps + 2)) + [10., 2.],
                               top_env_cls=CombinedEnvironment, 
                               lower_env_cls=HierarchicallyCombinedEnvironment, 
                               top_env_cfg=boxes_cfg, 
                               lower_env_cfg=arm_sticks_object_cfg, 
                               fun_m_lower= lambda m:m,
                               fun_s_lower=lambda m,s:s[-2:]+s[-2:]+s[-2:]+s[-2:]+s[-2:]+s[-2:]+s[-2:]+s[-2:]+s[-2:]+s[-2:],
                               fun_s_top=lambda m,s_lower,s: s_lower + s)
        
        denv_cfg = dict(env_cfg=static_env_cfg,
                        env_cls=HierarchicallyCombinedEnvironment,
                        m_mins=[-1.] * 4 * 3, 
                        m_maxs=[1.] * 4 * 3, 
                        s_mins=[-1.] * (3 * 3) + [-1.5] * 3 + [0.] * 3 + [-1.5] * 3 + [0.] * 3 + [-2., -2.] + [0., 0.],
                        s_maxs=[1.] * (3 * 3) + [1.5, 1.5] * 3 + [1.5, 1.5] * 3 + [2., 2.] + [10., 0.3],
                        n_bfs = 2,
                        n_motor_traj_points=3, 
                        n_sensori_traj_points=3, 
                        move_steps=move_steps, 
                        n_dynamic_motor_dims=4,
                        n_dynamic_sensori_dims=7, 
                        max_params=max_params,
                        motor_traj_type="DMP", 
                        sensori_traj_type="samples",
                        optim_initial_position=False, 
                        optim_end_position=True, 
                        default_motor_initial_position=[0.]*4, 
                        default_motor_end_position=[0.]*4,
                        default_sensori_initial_position=[0., 1., 0., 0., -0.85, 0.35, 1.2, 0.7], 
                        default_sensori_end_position=[0., 1., 0., 0., -0.85, 0.35, 1.2, 0.7],
                        gui=gui)
            
        
        DynamicEnvironment.__init__(self, **denv_cfg)
        
        
        