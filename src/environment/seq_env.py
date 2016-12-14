import numpy as np

from explauto.environment.environment import Environment


class SeqEnvironment(Environment):
    
    def __init__(self, environment, env_cls, env_config, m_mins, m_maxs, env_seq_s_mins, env_seq_s_maxs, n, combined_s):
        
        
        self.env = env_cls(**env_config)
        self.n_params_env = len(self.env.conf.m_dims)
        self.n = n
        self.combined_s = combined_s
        
        
        Environment.__init__(self, **environment)
        
    def rest_position(self):
        return list([self.env.rest_position] * self.n)
        
    def rest_params(self):
        return list([self.env.rest_params] * self.n)
    
    def compute_motor_command(self, m_ag):
        return np.reshape(self.env.compute_motor_command(np.reshape(m_ag, (self.n, len(m_ag)/self.n))), (len(m_ag),))
        
    def get_m_env(self, m, k):
        if len(np.shape(m)) == 1:
            return m[k * self.n_params_env: (k+1) * self.n_params_env]
        else:
            return m[:, k * self.n_params_env: (k+1) * self.n_params_env]
        
    def compute_sensori_effect(self, m):
        result = []
        for k in range(self.n):
            sk = list(self.env.update(self.get_m_env(m,k)))
            result += [sk]
        return self.combined_s(result)
    