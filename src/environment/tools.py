import time
import numpy as np
import matplotlib.pyplot as plt

from explauto.utils import bounds_min_max
from explauto.environment.environment import Environment
from explauto.environment.simple_arm.simple_arm import joint_positions
from grip_arm import GripArmEnvironment
from combined_env import CombinedEnvironment, HierarchicallyCombinedEnvironment
from dynamic_env import DynamicEnvironment


class Stick(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 length, handle_tol, handle_noise, rest_state, perturbation=None):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.length = length
        self.handle_tol = handle_tol
        self.handle_tol_sq = handle_tol * handle_tol
        self.handle_noise = handle_noise
        self.rest_state = rest_state
        self.perturbation = perturbation
        
        if self.perturbation == "BrokenTool1":
            self.length_breakpoint = 0.5
            self.angle_breakpoint = np.pi * 0.5
        
        self.reset()


    def reset(self):
        self.held = False
        self.handle_pos = np.array(self.rest_state[0:2])
        self.angle = self.rest_state[2]
        self.compute_end_pos()
        self.logs = []
        
    def compute_end_pos(self):
        if self.perturbation == "BrokenTool1":
            a = np.pi * self.angle
            breakpoint = [self.handle_pos[0] + np.cos(a) * self.length * self.length_breakpoint, 
                            self.handle_pos[1] + np.sin(a) * self.length * self.length_breakpoint]
            self.end_pos = [breakpoint[0] + np.cos(a + self.angle_breakpoint) * (self.length * (1. - self.length_breakpoint)), 
                            breakpoint[1] + np.sin(a + self.angle_breakpoint) * (self.length * (1. - self.length_breakpoint))]
        else:
            a = np.pi * self.angle
            self.end_pos = [self.handle_pos[0] + np.cos(a) * self.length, 
                            self.handle_pos[1] + np.sin(a) * self.length]
                
        
        
    def compute_motor_command(self, m):
        #return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        return m

    def compute_sensori_effect(self, m):
        hand_pos = m[0:2]
        hand_angle = m[2]
        gripper_change = m[3]
        
        if not self.held:
            if gripper_change == 1. and (hand_pos[0] - self.handle_pos[0]) ** 2. + (hand_pos[1] - self.handle_pos[1]) ** 2. < self.handle_tol_sq:
                self.handle_pos = hand_pos
                self.angle = np.mod(hand_angle + self.handle_noise * np.random.randn() + 1, 2) - 1
                self.compute_end_pos()
                self.held = True
        else:
            if gripper_change == 0:
                self.handle_pos = hand_pos
                self.angle = np.mod(hand_angle + self.handle_noise * np.random.randn() + 1, 2) - 1
                self.compute_end_pos()
            else:
                self.held = False
        
        #print "Stick log added"
        self.logs.append([self.handle_pos, 
                          self.angle, 
                          self.end_pos, 
                          self.held])
        #print "Tool hand_pos:", hand_pos, "hand_angle:", hand_angle, "gripper_change:", gripper_change, "self.handle_pos:", self.handle_pos, "self.angle:", self.angle, "self.held:", self.held 
        return list(self.end_pos) # Tool pos
    
    def plot(self, ax, i, **kwargs_plot):
        #print self.logs
        handle_pos = self.logs[i][0]
        end_pos = self.logs[i][2]
        
        
        if self.perturbation == "BrokenTool1":
            a = np.pi * self.logs[i][1]
            breakpoint = [handle_pos[0] + np.cos(a) * self.length * self.length_breakpoint, 
                            handle_pos[1] + np.sin(a) * self.length * self.length_breakpoint]
        
            ax.plot([handle_pos[0], breakpoint[0]], [handle_pos[1], breakpoint[1]], '-b', lw=6)
            ax.plot([breakpoint[0], end_pos[0]], [breakpoint[1], end_pos[1]], '-b', lw=6)
        else:
            ax.plot([handle_pos[0], end_pos[0]], [handle_pos[1], end_pos[1]], '-b', lw=6)
        ax.plot(handle_pos[0], handle_pos[1], 'or', ms=12)
        ax.plot(end_pos[0], end_pos[1], 'og', ms=12)
                

class SceneObject(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 object_shape, object_tol1, object_tol2, pair_noise, rest_state):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.object_shape = object_shape
        self.object_tol1 = object_tol1
        self.object_tol1_sq = object_tol1 * object_tol1
        self.object_tol2 = object_tol2
        self.object_tol2_sq = object_tol2 * object_tol2
        self.pair_noise = pair_noise
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
        if self.conf.m_ndims == 5 and (self.move == 2 or ((not self.move == 1) and (m[2] - self.pos[0]) ** 2 + (m[3] - self.pos[1]) ** 2 < self.object_tol2_sq)):
            self.pos = m[2:4]
            self.move = 2
            #print "OBJECT PUSHED BY TOOL 2!"
            
        self.logs.append([self.pos,
                          self.move])
        return list(self.pos)
    
    def pair_help(self, m_traj, s):
        last_pos = s[-1]
        new_pos = list(last_pos)
        help_signal = m_traj[-1][self.conf.m_ndims-1]
        if self.move == 1: # if last tool that pushed object was tool1
            #print "help_signal", help_signal
            if help_signal > 0.:
                z = last_pos[0] + last_pos[1]*1j
                r = np.abs(z)
                if r > 1.2 and r < 1.5:
                    theta = np.angle(z)
                    if theta > -0.262 and theta < 0.785:#-15;45
                        new_pos[0] = 1.75 + np.random.randn() * self.pair_noise[1]
                        new_pos[1] = 0. + np.random.randn() * self.pair_noise[1]
                    elif theta > 0.785 and theta < 1.833:#45;105
                        new_pos[0] = 0. + np.random.randn() * self.pair_noise[0]
                        new_pos[1] = 1.75 + np.random.randn() * self.pair_noise[0]
            self.logs[-1][0] = new_pos
        return [help_signal] + new_pos
    
    def update(self, m_traj, reset=False, log=True, batch=False):
        if reset:
            self.reset()
        s = []
        #print "SceneObject update m_traj", m_traj
        for m in m_traj:
            #print "SceneObject update m", m
            s.append(self.one_update(m, log))
        return self.pair_help(m_traj, s)
    
    def plot(self, ax, i, **kwargs_plot):
        pos = self.logs[i][0]        
        if self.object_shape == "rectangle":
            rectangle = plt.Rectangle((pos[0] - self.object_tol1 / 2., pos[1] - self.object_tol1 / 2.), self.object_tol1, self.object_tol1, fc='y')
            ax.add_patch(rectangle)
        elif self.object_shape == "circle":
            circle = plt.Circle((pos[0], pos[1]), self.object_tol1, fc='y')
            ax.add_patch(circle)            
        else:
            raise NotImplementedError       
        
        
class Box(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs, box_m_mins, box_m_maxs):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        
        self.box_m_mins = box_m_mins
        self.box_m_maxs = box_m_maxs
        self.center = (np.array(self.box_m_mins) + np.array(self.box_m_maxs)) / 2.
        self.reset()
        
    def reset(self):        
        self.logs = []
        
    def compute_motor_command(self, m):
        #return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        return m

    def compute_sensori_effect(self, m):
        d = np.linalg.norm(np.array(m[0:2]) - self.center)
        if m[0] >= self.box_m_mins[0] and m[0] <= self.box_m_maxs[0] and m[1] >= self.box_m_mins[1] and m[1] <= self.box_m_maxs[1]:            
            full = 1
        else:
            full = 0
        self.logs.append([full, d])
        return [full, d] # object pos + hand pos + gripper state + tool1 end pos + tool2 end pos
    
    def plot(self, ax, i, **kwargs_plot): 
        if i >= len(self.logs):
            i = len(self.logs) - 1       
        if self.logs[i][0]:
            fc = "none"#"k"
        else:
            fc = "none"     
        rectangle = plt.Rectangle((self.box_m_mins[0], self.box_m_mins[1]), 
                                  self.box_m_maxs[0] - self.box_m_mins[0], 
                                  self.box_m_maxs[1] - self.box_m_mins[1], 
                                  fc=fc, alpha=0.5, lw=4)
        ax.add_patch(rectangle)


class VocalSignal(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs, box_m_mins, box_m_maxs):
    
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
        
        self.box_m_mins = box_m_mins # In the box the signal will be 1., else -1.
        self.box_m_maxs = box_m_maxs

        self.reset()
        
    def reset(self):        
        self.logs = []
        
    def compute_motor_command(self, m):
        #return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        return m

    def compute_sensori_effect(self, m):
        if m[0] >= self.box_m_mins[0] and m[0] <= self.box_m_maxs[0] and m[1] >= self.box_m_mins[1] and m[1] <= self.box_m_maxs[1]:            
            s = 1
        else:
            s = -1.
        self.logs.append([s])
        return [s]
    
    def plot(self, ax, i, **kwargs_plot): 
        pass
        

class ToolEnvironment(DynamicEnvironment):
    def __init__(self, env_type="small_tool", move_steps=50, max_params=None, perturbation=None, gui=False):

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
                         length=0.5, 
                         handle_tol=0.2, 
                         handle_noise=0.02, 
                         rest_state=[-0.5, 0, 0.75],
                         perturbation=perturbation)
        
        if env_type == "both_tools":
            stick2_cfg = dict(m_mins=[-1, -1, -1, -1, -1], 
                             m_maxs=[1, 1, 1, 1, 1], 
                             s_mins=[-2, -2], 
                             s_maxs=[2, 2],
                             length=1., 
                             handle_tol=0.4, 
                             handle_noise=0.1, 
                             rest_state=[0.5, 0., 0.25])
            
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
        
        elif env_type == "small_tool":
            arm_stick_cfg = dict(m_mins=list([-1.] * 4), # 3DOF + gripper
                                 m_maxs=list([1.] * 4),
                                 s_mins=list([-2.] * 5),
                                 s_maxs=list([2.] * 5),
                                 top_env_cls=Stick, 
                                 lower_env_cls=GripArmEnvironment, 
                                 top_env_cfg=stick1_cfg, 
                                 lower_env_cfg=gripArm_cfg, 
                                 fun_m_lower= lambda m:[motor_perturbation(m_t[0:4]) for m_t in m],
                                 fun_s_lower=lambda m,s:s,  # (hand pos + hand angle + gripper_change + gripper state)
                                 fun_s_top=lambda m,s_lower,s:[s_l[0:2] + [s_l[4]] + s_ for s_l, s_ in zip(s_lower, s)]) # from s: Tool1 end pos + Tool2 end pos  from m: hand_pos + gripper state
        else:
            raise NotImplementedError
        
        vocal_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0.],
                     s_maxs = [1.],
                     box_m_mins = [0., 0.], 
                     box_m_maxs = [1., 1.]
                     )
        
        arm_sticks_vocal_cfg = dict(
                        s_mins = arm_stick_cfg['s_mins'] + vocal_cfg['s_mins'],
                        s_maxs = arm_stick_cfg['s_mins'] + vocal_cfg['s_maxs'],
                        envs_cls = [HierarchicallyCombinedEnvironment, VocalSignal], 
                        envs_cfg = [arm_stick_cfg, vocal_cfg], 
                        combined_s = lambda s:s
                        )
        
        if env_type == "both_tools":
            object_cfg = dict(m_mins = list([-1.] * 5), 
                              m_maxs = list([1.] * 5), 
                              s_mins = [-1., -1., -1.], # help signal + new pos
                              s_maxs = [1., 1., 1.],
                              object_shape = "rectangle", 
                              object_tol1 = 0.2, 
                              object_tol2 = 0.4, 
                              pair_noise = [0.2, 0.05],
                              rest_state = [-0.75, 1.])
            
            arm_sticks_vocal_object_cfg = dict(
                                               m_mins=arm_stick_cfg['m_mins'] + vocal_cfg['m_mins'],
                                               m_maxs=arm_stick_cfg['m_maxs'] + vocal_cfg['m_maxs'],
                                               s_mins=list([-2.] * (7 * move_steps + 3)),
                                               s_maxs=list([2.] * (7 * move_steps + 3)), # (hand pos + gripper state + tool1 end pos + tool2 end pos) traj + help signal + last object pos = 7 x samples + 3
                                               top_env_cls=SceneObject, 
                                               lower_env_cls=CombinedEnvironment, 
                                               top_env_cfg=object_cfg, 
                                               lower_env_cfg=arm_sticks_vocal_cfg, 
                                               fun_m_lower= lambda m:m,
                                               fun_s_lower=lambda m,s:s[3:],
                                               fun_s_top=lambda m,s_lower,s: [si[:7] for si in s_lower] + s)
        
        elif env_type == "small_tool":   
            object_cfg = dict(m_mins = list([-1.] * 3), 
                              m_maxs = list([1.] * 3), 
                              s_mins = [-1., -1., -1.], # help signal + new pos
                              s_maxs = [1., 1., 1.],
                              object_shape = "rectangle", 
                              object_tol1 = 0.2, 
                              object_tol2 = 0.4, 
                              pair_noise = [0.2, 0.05],
                              rest_state = [-0.75, 1.])
            
            arm_sticks_vocal_object_cfg = dict(
                                               m_mins=arm_stick_cfg['m_mins'] + vocal_cfg['m_mins'],
                                               m_maxs=arm_stick_cfg['m_maxs'] + vocal_cfg['m_maxs'],
                                               s_mins=list([-2.] * (5 * move_steps + 3)),
                                               s_maxs=list([2.] * (5 * move_steps + 3)), # (hand pos + gripper state + tool1 end pos + tool2 end pos) traj + help signal + last object pos = 7 x samples + 3
                                               top_env_cls=SceneObject, 
                                               lower_env_cls=CombinedEnvironment, 
                                               top_env_cfg=object_cfg, 
                                               lower_env_cfg=arm_sticks_vocal_cfg, 
                                               fun_m_lower= lambda m:m,
                                               fun_s_lower=lambda m,s:s[3:],
                                               fun_s_top=lambda m,s_lower,s: [si[:5] for si in s_lower] + s)
             
        else:
            raise NotImplementedError
        
        box1_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0., 0.],
                     s_maxs = [1., 1.],
                     box_m_mins = [-1.5, -0.25], 
                     box_m_maxs = [-1., 0.25]
                     )
        
        box2_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0., 0.],
                     s_maxs = [1., 1.],
                     box_m_mins = [-0.25, 1.5], 
                     box_m_maxs = [0.25, 2.]
                     )
        
        box3_cfg = dict(
                     m_mins = [-1., -1.],
                     m_maxs = [1., 1.],
                     s_mins = [0., 0.],
                     s_maxs = [1., 1.],
                     box_m_mins = [1.5, -0.25], 
                     box_m_maxs = [2., 0.25]
                     )
        
        def combined_boxes(s):
            if s[0]:
                b = -0.25
            elif s[2]:
                b = 0.25
            elif s[4]:
                b = 0.75
            else:
                b = -0.75
            d = min([s[1], s[3], s[5]])
            return [b, d]
        
        boxes_cfg = dict(
                        s_mins = list([-1., 0]),
                        s_maxs = list([1., 1.]),
                        envs_cls = [Box, Box, Box], 
                        envs_cfg = [box1_cfg, box2_cfg, box3_cfg], 
                        combined_s = combined_boxes
                        )
#         
#         boxes_cfg = dict(
#                         s_mins = list([0.] * 8),
#                         s_maxs = list([1.] * 8),
#                         envs_cls = [Box, Box, Box], 
#                         envs_cfg = [box1_cfg, box2_cfg, box3_cfg], 
#                         combined_s = lambda s:s
#                         )
                 
        
        if env_type == "both_tools":
            static_env_cfg = dict(m_mins=arm_stick_cfg['m_mins'],
                           m_maxs=arm_stick_cfg['m_maxs'],
                           s_mins=list([-2.] * (7 * move_steps + 9)),
                           s_maxs=list([2.] * (7 * move_steps + 9)),
                           top_env_cls=CombinedEnvironment, 
                           lower_env_cls=HierarchicallyCombinedEnvironment, 
                           top_env_cfg=boxes_cfg, 
                           lower_env_cfg=arm_sticks_vocal_object_cfg, 
                           fun_m_lower= lambda m:m,
                           fun_s_lower=lambda m,s:s[-2:]+s[-2:]+s[-2:],
                           fun_s_top=lambda m,s_lower,s: s_lower + s)
            
            denv_cfg = dict(env_cfg=static_env_cfg,
                            env_cls=HierarchicallyCombinedEnvironment,
                            m_mins=[-1.] * 4 * 3 + vocal_cfg['m_mins'], 
                            m_maxs=[1.] * 4 * 3 + vocal_cfg['m_maxs'], 
                            s_mins=[-1.] * (3 * 3) + [-1.5] * (2 * 3) + [-2.] * (2 * 3) + [-1.] + [-2., -2.] + [-1., 0.], 
                            s_maxs=[1.] * (3 * 3) + [1.5] * (2 * 3) + [2.] * (2 * 3) + [1.] + [2., 2.] + [1., 1.],
                            n_motor_traj_points=3, 
                            n_sensori_traj_points=3, 
                            move_steps=move_steps, 
                            n_dynamic_motor_dims=4,
                            n_dynamic_sensori_dims=7, 
                            max_params=max_params,
                            motor_traj_type="DMP", 
                            sensori_traj_type="samples",
                            optim_initial_position=False, 
                            optim_end_position=False, 
                            default_motor_initial_position=[0.]*4, 
                            default_motor_end_position=[0.]*4,
                            default_sensori_initial_position=[0., 1., 0., 0., -0.85, 0.35, 1.2, 0.7], 
                            default_sensori_end_position=[0., 1., 0., 0., -0.85, 0.35, 1.2, 0.7],
                            gui=gui)
            
        elif env_type == "small_tool":  
            static_env_cfg = dict(m_mins=arm_stick_cfg['m_mins'],
                           m_maxs=arm_stick_cfg['m_maxs'],
                           s_mins=list([-2.] * (5 * move_steps + 9)),
                           s_maxs=list([2.] * (5 * move_steps + 9)),
                           top_env_cls=CombinedEnvironment, 
                           lower_env_cls=HierarchicallyCombinedEnvironment, 
                           top_env_cfg=boxes_cfg, 
                           lower_env_cfg=arm_sticks_vocal_object_cfg, 
                           fun_m_lower= lambda m:m,
                           fun_s_lower=lambda m,s:s[-2:]+s[-2:]+s[-2:],
                           fun_s_top=lambda m,s_lower,s: s_lower + s)
            
            denv_cfg = dict(env_cfg=static_env_cfg,
                            env_cls=HierarchicallyCombinedEnvironment,
                            m_mins=[-1.] * 4 * 3 + vocal_cfg['m_mins'], 
                            m_maxs=[1.] * 4 * 3 + vocal_cfg['m_maxs'], 
                            s_mins=[-1.] * (3 * 3) + [-1.5] * (2 * 3) + [-1.] + [-2., -2.] + [-1., 0.], 
                            s_maxs=[1.] * (3 * 3) + [1.5] * (2 * 3) + [1.] + [2., 2.] + [1., 1.],
                            n_motor_traj_points=3, 
                            n_sensori_traj_points=3, 
                            move_steps=move_steps, 
                            n_dynamic_motor_dims=4,
                            n_dynamic_sensori_dims=5, 
                            max_params=max_params,
                            motor_traj_type="DMP", 
                            sensori_traj_type="samples",
                            optim_initial_position=False, 
                            optim_end_position=False, 
                            default_motor_initial_position=[0.]*4, 
                            default_motor_end_position=[0.]*4,
                            default_sensori_initial_position=[0., 1., 0., 0., -0.85, 0.35], 
                            default_sensori_end_position=[0., 1., 0., 0., -0.85, 0.35],
                            gui=gui) 
        else:
            raise NotImplementedError
        
        DynamicEnvironment.__init__(self, **denv_cfg)
        
        
        