import matplotlib.pyplot as plt

from numpy.linalg import norm
from numpy import array, mean

from explauto.environment.simple_arm import SimpleArmEnvironment


class ArmEnv(SimpleArmEnvironment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                 length_ratio, pois, dist, triggers, combine_triggers, default_trigger):

        SimpleArmEnvironment.__init__(self, m_mins, m_maxs, s_mins, s_maxs, length_ratio, 0.)
        self.pois = pois
        self.dist = dist
        self.triggers = triggers
        self.combine_triggers = combine_triggers
        self.default_trigger = default_trigger

    def compute_sensori_effect(self, joint_pos_env):
        
        s = list(SimpleArmEnvironment.compute_sensori_effect(self, joint_pos_env))
        t = []
        for i in range(len(self.pois)):
            if norm(array(s) - array(self.pois[i])) < self.dist:
                t += [self.triggers[i]]
            else:
                t += [self.default_trigger]
        s += list(self.combine_triggers(t))
        return s

    def trajectory(self, m): return m
    
    def plot(self, ax, m_, **kwargs_plot):
        SimpleArmEnvironment.plot_arm(self, ax, m_, **kwargs_plot)
        for i in range(len(self.pois)):
            c = plt.Circle((self.pois[i][0],self.pois[i][1]),self.dist,color='b',fill=False)
            ax.add_artist(c)
            
            

if __name__ == '__main__':
    
    saConfig = dict(
        m_mins = [-3, -3, -3],
        m_maxs = [3, 3, 3],
        s_mins = [-1,-1],
        s_maxs = [1, 1],
        length_ratio = 2,
        pois = [[0.5, 0.5],
                [0.5, -0.5]],
        dist = 0.1,
        triggers = [[500, 900], 
                    [300, 1700]],
        combine_triggers = lambda l:[0,0] if [pt for pt in l if pt <> [0,0]] == [] else mean([pt for pt in l if pt <> [0,0]],axis=0),
        default_trigger = [0, 0]
        )
    
    saEnv = ArmEnv(**saConfig)
    
    m = [0.1,1.4,0.8]
    print "m=", m, "s=", saEnv.compute_sensori_effect(m)
    saEnv.plot(plt.subplot(), m)
    plt.show()