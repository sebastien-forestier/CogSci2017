import numpy as np
import matplotlib.pyplot as plt

from explauto.utils import bounds_min_max
from explauto.environment.simple_arm.simple_arm import forward, joint_positions
from explauto.environment.environment import Environment


class GripArmEnvironment(Environment):
    use_process = True

    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 lengths, angle_shift, rest_state):
        
        Environment.__init__(self, m_mins, m_maxs, s_mins, s_maxs)

        self.lengths = lengths
        self.angle_shift = angle_shift
        self.rest_state = rest_state
        self.reset()
        
    def reset(self):
        self.gripper = self.rest_state[3]
        self.logs = []
        
    def compute_motor_command(self, m):
        return bounds_min_max(m, self.conf.m_mins, self.conf.m_maxs)
        #return m

    def compute_sensori_effect(self, m):
        a = self.angle_shift + np.cumsum(np.array(m[:-1]))
        a_pi = np.pi * a 
        hand_pos = np.array([np.sum(np.cos(a_pi)*self.lengths), np.sum(np.sin(a_pi)*self.lengths)])
        if m[-1] >= 0.:
            new_gripper = 1. 
        else:
            new_gripper = -1.
        gripper_change = (self.gripper - new_gripper) / 2.
        self.gripper = new_gripper
        angle = np.mod(a[-1] + 1, 2) - 1
        self.logs.append(m)
        return [hand_pos[0], hand_pos[1], angle, gripper_change, self.gripper]
    
    def plot(self, ax, i, **kwargs_plot):
        m = self.logs[i]
        angles = m[:-1]
        angles[0] += self.angle_shift
        x, y = joint_positions(angles, self.lengths, 'std')
        x, y = [np.hstack((0., a)) for a in x, y]
        ax.plot(x, y, 'grey', lw=4, **kwargs_plot)
        ax.plot(x[0], y[0], 'ok', ms=12)
        for i in range(len(self.lengths)-1):
            ax.plot(x[i+1], y[i+1], 'ok', ms=12)
        ax.plot(x[-1], y[-1], 'or', ms=4)
        ax.axis([-1.6, 2.1, -1., 2.1])        

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        self.plot_gripper(ax, x[-1], y[-1], np.cumsum(m[:-1]), m[-1] >= 0.)
        
    def plot_gripper(self, ax, x, y, angle, gripper_open):
        if gripper_open:
            ax.plot(x, y, 'o', markerfacecolor='none', markeredgewidth=6, markeredgecolor='g', ms=26)
        else:
            ax.plot(x, y, 'og', ms=12)
        
        

if __name__ == '__main__':
    
    m_mins = [-1, -1, -1, -1]
    m_maxs = [1, 1, 1, 1]
    s_mins = [-1, -1, -1, -1]
    s_maxs = [1, 1, 1, 1]
    
    lengths = [0.5, 0.3, 0.2]
    noise = 0.
    rest_state = [0., 0., 0., 0.]
    
    arm = GripArmEnvironment(m_mins, m_maxs, s_mins, s_maxs, lengths, noise, rest_state)
    
    m = rest_state
    arm.update(m)
    
    ax = plt.subplot()
    plt.show(block=False)
    
    for i in range(100):
        plt.cla()
        f = float(i) / 100.
        arm.plot(ax, [-0.5+f, -0.5+f, f, 0.5 - f/2])
        plt.draw()
        
    plt.show()
    