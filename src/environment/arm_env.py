import numpy as np

from explauto.utils import bounds_min_max
from explauto.environment.environment import Environment
from explauto.environment.simple_arm.simple_arm import joint_positions
from ..dmp.mydmp import MyDMP


class ArmEnvironment(Environment):
    use_process = True

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 lengths, angle_shift, rest_state, n_dmps=3, n_bfs=6, timesteps=50, gui=False):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.lengths = lengths
        self.angle_shift = angle_shift
        self.rest_state = rest_state
        self.reset()
        self.gui = gui
        
        # DMP PARAMETERS
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.timesteps = timesteps
        self.max_params = np.array([300.] * self.n_bfs * self.n_dmps + [1.] * self.n_dmps)
        self.motor_dmp = MyDMP(n_dmps=self.n_dmps, n_bfs=self.n_bfs, timesteps=self.timesteps, max_params=self.max_params)
        
        
    def reset(self):
        #print "reset gripper"
        self.logs = []
        
    def compute_motor_command(self, m):
        return m

    def compute_traj(self, m):
        return bounds_min_max(self.motor_dmp.trajectory(np.array(m) * self.max_params), self.n_dmps * [-1.], self.n_dmps * [1.])
        
    def compute_sensori_effect(self, m):
        m_traj = self.compute_traj(m)
        s = []
        for m in m_traj:
            a = self.angle_shift + np.cumsum(np.array(m))
            a_pi = np.pi * a 
            hand_pos = np.array([np.sum(np.cos(a_pi)*self.lengths), np.sum(np.sin(a_pi)*self.lengths)])
            angle = np.mod(a[-1] + 1, 2) - 1
            s.append([hand_pos[0], hand_pos[1], angle]) 
            self.logs.append(m)
        return s
    
    
    def plot_step(self, ax, i, **kwargs_plot):
        m = self.logs[i]
        angles = m
        angles[0] += self.angle_shift
        x, y = joint_positions(angles, self.lengths, 'std')
        x, y = [np.hstack((0., a)) for a in x, y]
        ax.plot(x, y, 'grey', lw=4, **kwargs_plot)
        ax.plot(x[0], y[0], 'ok', ms=12, **kwargs_plot)
        for j in range(len(self.lengths)-1):
            ax.plot(x[j+1], y[j+1], 'ok', ms=12, **kwargs_plot)
        ax.plot(x[-1], y[-1], 'or', ms=12, **kwargs_plot)
        #ax.axis([-1.6, 2.1, -1., 2.1])        

