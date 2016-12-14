import numpy as np
import time
import matplotlib.pyplot as plt

from explauto.models.dmp import DmpPrimitive
from explauto.utils.utils import bounds_min_max
from explauto.environment.environment import Environment


class DynamicEnvironment(Environment):
    def __init__(self, env_cls, env_cfg, 
                 m_mins, m_maxs, s_mins, s_maxs,
                 n_bfs, n_motor_traj_points, n_sensori_traj_points, move_steps, 
                 n_dynamic_motor_dims, n_dynamic_sensori_dims, max_params,
                 motor_traj_type="DMP", sensori_traj_type="DMP", 
                 optim_initial_position=True, optim_end_position=True, default_motor_initial_position=None, default_motor_end_position=None,
                 default_sensori_initial_position=None, default_sensori_end_position=None,
                 gui=False):
        
        self.env = env_cls(**env_cfg)
        
        self.n_bfs = n_bfs
        self.n_motor_traj_points = n_motor_traj_points
        self.n_sensori_traj_points = n_sensori_traj_points 
        self.move_steps = move_steps
        self.n_dynamic_motor_dims = n_dynamic_motor_dims
        self.n_dynamic_sensori_dims = n_dynamic_sensori_dims
        self.max_params = max_params
        self.motor_traj_type = motor_traj_type 
        self.sensori_traj_type = sensori_traj_type 
        self.optim_initial_position = optim_initial_position
        self.optim_end_position = optim_end_position
        self.gui = gui
        if self.gui:
            self.ax = plt.subplot()
            plt.gcf().set_size_inches(12., 12., forward=True)
            plt.gca().set_aspect('equal')
            plt.show(block=False)
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)
                
        if self.motor_traj_type == "DMP":
            self.init_motor_DMP(optim_initial_position, optim_end_position, default_motor_initial_position, default_motor_end_position)
        else:
            raise NotImplementedError
            
        if self.sensori_traj_type == "DMP":
            self.init_sensori_DMP(optim_initial_position, optim_end_position, default_sensori_initial_position, default_sensori_end_position)
        elif self.sensori_traj_type == "samples":
            self.samples = np.array(np.linspace(-1, self.move_steps-1, self.n_sensori_traj_points + 1), dtype=int)[1:]
        else:
            raise NotImplementedError
            
    def reset(self):
        self.env.reset()
            
    def init_motor_DMP(self, optim_initial_position=True, optim_end_position=True, default_motor_initial_position=None, default_motor_end_position=None):
        default = np.zeros(self.n_dynamic_motor_dims * (self.n_bfs + 2))
        if not optim_initial_position:
            default[:self.n_dynamic_motor_dims] = default_motor_initial_position
            dims_optim = [False] * self.n_dynamic_motor_dims
        else:
            dims_optim = [True] * self.n_dynamic_motor_dims
        dims_optim += [True] * (self.n_dynamic_motor_dims * self.n_bfs)
        if not optim_end_position:
            default[-self.n_dynamic_motor_dims:] = default_motor_end_position
            dims_optim += [False] * self.n_dynamic_motor_dims
        else:
            dims_optim += [True] * self.n_dynamic_motor_dims
        self.motor_dmp = DmpPrimitive(self.n_dynamic_motor_dims, 
                                    self.n_bfs, 
                                    dims_optim, 
                                    default, 
                                    type='discrete',
                                    timesteps=self.move_steps)
            
    def init_sensori_DMP(self, optim_initial_position=True, optim_end_position=True, default_sensori_initial_position=None, default_sensori_end_position=None):
        default = np.zeros(self.n_dynamic_sensori_dims * (self.n_sensori_traj_points + 2))
        if not optim_initial_position:
            default[:self.n_dynamic_sensori_dims] = default_sensori_initial_position
            dims_optim = [False] * self.n_dynamic_sensori_dims
        else:
            dims_optim = [True] * self.n_dynamic_sensori_dims
        dims_optim += [True] * (self.n_dynamic_sensori_dims * self.n_sensori_traj_points)
        if not optim_end_position:
            default[-self.n_dynamic_sensori_dims:] = default_sensori_end_position
            dims_optim += [False] * self.n_dynamic_sensori_dims
        else:
            dims_optim += [True] * self.n_dynamic_sensori_dims
        self.sensori_dmp = DmpPrimitive(self.n_dynamic_sensori_dims, 
                                        self.n_sensori_traj_points, 
                                        dims_optim, 
                                        default, 
                                        type='discrete',
                                        timesteps=self.move_steps)
    
    def compute_motor_command(self, m_ag):  
        m_ag = bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)
        if self.motor_traj_type == "DMP":
            dyn_idx = range(self.n_dynamic_motor_dims * self.n_motor_traj_points)
            #print "m params", m_ag[dyn_idx] * self.max_params
            m_dyn = self.motor_dmp.trajectory(m_ag[dyn_idx] * self.max_params)
            #print "mov", m_dyn
            static_idx = range(self.n_dynamic_motor_dims * self.n_motor_traj_points, self.conf.m_ndims)
            m_static = m_ag[static_idx]
            m = [list(m_dyn_param) + list(m_static) for m_dyn_param in list(m_dyn)]
        else:
            raise NotImplementedError
        return m
    
    def compute_sensori_effect(self, m_traj):
        s = self.env.update(m_traj, log=False)
        y = np.array(s[:self.move_steps])
        if self.sensori_traj_type == "DMP":
            self.sensori_dmp.dmp.imitate_path(np.transpose(y))
            w = self.sensori_dmp.dmp.w.flatten()
            s_ag = list(w)    
        elif self.sensori_traj_type == "samples":
            w = y[self.samples,:]
            s_ag = list(np.transpose(w).flatten())
        else:
            raise NotImplementedError  
        s = s_ag + s[self.move_steps:] 
        if self.gui:
            #if abs(s[11] - (-0.85)) > 0.1: #Tool1
            #if s[-2] > 0: # One of the boxes
            self.plot()
        return bounds_min_max(s, self.conf.s_mins, self.conf.s_maxs)    
        
    def update(self, m_ag, reset=False, log=False, batch=False):
        return Environment.update(self, m_ag, reset=True, log=log, batch=batch)
        
    def plot(self, **kwargs):
        for i in range(self.move_steps):
            plt.cla()
            self.env.plot(self.ax, i, **kwargs)
#             plt.xlim([-1.3, 1.3])
#             plt.ylim([-0.2, 1.6])
            plt.xlim([-1.6, 1.6])
            plt.ylim([-0.5, 1.6])
#             plt.gca().set_xticklabels([])
#             plt.gca().set_yticklabels([])
#             plt.gca().yaxis.set_major_locator(plt.NullLocator())
#             plt.gca().yaxis.set_major_locator(plt.NullLocator())
#             plt.xlabel("")
#             plt.ylabel("")
            plt.draw()
            if False:
                if i in [16, 32, 49]:
                    plt.savefig('/home/sforesti/scm/PhD/cogsci2016/include/test-mvt-' + str(i) + '.pdf', format='pdf', dpi=1000, bbox_inches='tight')
        #time.sleep(1)

