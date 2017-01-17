import os
import numpy as np
import time

from numpy import array, hstack, float32, zeros, linspace, shape, mean, log2, transpose, sum, isnan

from oct2py import Oct2Py, Oct2PyError
from explauto.environment.environment import Environment
from explauto.utils import bounds_min_max
from explauto.models.dmp import DmpPrimitive
from ...dmp.mydmp import MyDMP

if not (os.environ.has_key('AVAKAS') and os.environ['AVAKAS']):
    import pyaudio

                       
                       
class DivaSynth:
    def __init__(self, sample_rate=11025):
        # sample rate setting not working yet
        self.diva_path = os.path.join(os.getenv("HOME"), 'software/DIVAsimulink/')
        assert os.path.exists(self.diva_path)
        self.octave = Oct2Py()
        self.restart_iter = 500
        self.init_oct()

    def init_oct(self):
        self.octave.addpath(self.diva_path)
        self.iter = 0
        
    def execute(self, art):
        try:
            self.aud, self.som, self.vt = self.octave.diva_synth(art, 'audsom')
        except:
            self.reboot()
            print "Warning: Oct2Py crashed, Oct2Py restarted"
            self.aud, self.som, self.vt = self.octave.diva_synth(art, 'audsom')
        self.add_iter()
        return self.aud, self.som, self.vt

    def sound_wave(self, art):
        wave = self.octave.diva_synth(art, 'sound')
        self.add_iter()
        return wave
    
    def add_iter(self):
        if self.iter >= self.restart_iter:
            self.restart()
        else:
            self.iter += 1
            
    def reboot(self):
        self.octave = Oct2Py()
        self.init_oct()
        
    def restart(self):
        self.octave.restart()
        self.init_oct()
        
    def stop(self):
        self.octave.exit()



class DivaEnvironment(Environment):
    def __init__(self, m_mins, m_maxs, s_mins, s_maxs,
                m_used,
                s_used,
                rest_position_diva,
                audio,
                diva_use_initial,
                diva_use_goal,
                used_diva,
                n_dmps_diva,
                n_bfs_diva,
                move_steps):
        
        self.m_mins = m_mins
        self.m_maxs = m_maxs 
        self.s_mins = s_mins
        self.s_maxs = s_maxs
        self.m_used = m_used
        self.s_used = s_used
        self.rest_position_diva = rest_position_diva
        self.audio = audio
        self.diva_use_initial = diva_use_initial
        self.diva_use_goal = diva_use_goal
        self.used_diva = used_diva
        self.n_dmps_diva = n_dmps_diva
        self.n_bfs_diva = n_bfs_diva
        self.move_steps = move_steps
    
        self.f0 = 1.
        self.pressure = 1.
        self.voicing = 1.
        
        if (os.environ.has_key('AVAKAS') and os.environ['AVAKAS']):
            self.audio = False
        
        if self.audio:            
            self.pa = pyaudio.PyAudio()
            self.stream = self.pa.open(format=pyaudio.paFloat32,
                                        channels=1,
                                        rate=11025,
                                        output=True)
            
        self.synth = DivaSynth()
        self.art = array([0.]*10 + [self.f0, self.pressure, self.voicing])   # 13 articulators is a constant from diva_synth.m in the diva source code
        
        self.max_params = []
        
        if self.diva_use_initial:
            self.max_params = self.max_params + [1.] * self.n_dmps_diva
            self.max_params = self.max_params + [300.] * self.n_bfs_diva * self.n_dmps_diva
        if self.diva_use_goal:
            self.max_params = self.max_params + [1.] * self.n_dmps_diva
        self.max_params = np.array(self.max_params)
        
        self.dmp = MyDMP(n_dmps=self.n_dmps_diva, n_bfs=self.n_bfs_diva, timesteps=self.move_steps, use_init=self.diva_use_initial, max_params=self.max_params)
        
        self.default_m = zeros(self.n_dmps_diva * self.n_bfs_diva + self.n_dmps_diva * self.diva_use_initial + self.n_dmps_diva * self.diva_use_goal)
        self.default_m_traj = self.compute_motor_command(self.default_m)
        self.default_sound = self.synth.execute(self.art.reshape(-1,1))[0]
        self.default_formants = None
        self.default_formants = self.compute_sensori_effect(self.default_m_traj)
        
        Environment.__init__(self, self.m_mins, self.m_maxs, self.s_mins, self.s_maxs)

    def compute_motor_command(self, m_ag):
        return bounds_min_max(self.trajectory(m_ag), self.m_mins, self.m_maxs)


    def compute_sensori_effect(self, m_env):
        #print "compute_se", m_env
        if len(array(m_env).shape) == 1:
            self.art[self.m_used] = m_env
            #print self.art
            
            if m_env == self.default_m:
                res = self.default_sound
            else:
                res = self.synth.execute(2.*(self.art.reshape(-1,1)))[0]
            #print "compute_se result", res[self.s_used].flatten()
            formants = log2(transpose(res[self.s_used]))
            formants[isnan(formants)] = 0.
            return formants
        else:
            
            if self.default_formants is not None and (m_env == self.default_m_traj).all():
                return self.default_formants
            else:
                self.art_traj = zeros((13, array(m_env).shape[0]))
                #self.art_traj = zeros((13, array(m_env).shape[0]))
                self.art_traj[10, :] = self.f0
                self.art_traj[11, :] = self.pressure
                self.art_traj[12, :] = self.voicing
                self.art_traj[self.m_used,:] = transpose(m_env)
                
                res = self.synth.execute(2.*(self.art_traj))[0]
                
                #res = self.synth.execute(np.arctanh(self.art_traj))[0]
                
                #if isnan(sum(log2(transpose(res[self.s_used,:])))):
                #    print "diva NaN:"
    #                 print "m_env", m_env
    #                 print "self.art_traj", self.art_traj, 
    #                 print "res", res, 
    #                 print "formants", log2(transpose(res[self.s_used,:]))
                formants = log2(transpose(res[self.s_used,:]))
                formants[isnan(formants)] = 0.
                
                return formants

    def rest_params(self):
        dims = self.n_dmps_diva*self.n_bfs_diva
        if self.diva_use_initial:
            dims += self.n_dmps_diva
        if self.diva_use_goal:
            dims += self.n_dmps_diva
        rest = zeros(dims)
        if self.diva_use_initial:
            rest[:self.n_dmps_diva] = self.rest_position_diva
        if self.diva_use_goal:
            rest[-self.n_dmps_diva:] = self.rest_position_diva
        return rest
    
    
    def trajectory(self, m):
        y = self.dmp.trajectory(np.array(m) * self.max_params)
        if len(y) > self.move_steps: 
            ls = linspace(0,len(y)-1,self.move_steps)
            ls = array(ls, dtype='int')
            y = y[ls]
        #print "m diva", m, "traj diva", y
        return y
        
        
    def update(self, mov, audio=True):
        s = Environment.update(self, mov)
        
        if self.audio and audio:
            sound = self.sound_wave(self.art_traj)
            self.stream.write(sound.astype(float32).tostring())
            #time.sleep(1)
            #print "Sound sent", sound, len(sound)
        return s    
        #return [s_[0] for s_ in s] + [s_[1] for s_ in s] 
    
#         if len(shape(array(s))) == 1:
#             return s
#         else:
#             s = list(mean(array(s), axis=0))
#             #print "Diva s=", s
#             return s
#         
        
    def sound_wave(self, art_traj, power = 2.):
        synth_art = self.art.reshape(1, -1).repeat(len(art_traj.T), axis=0)
        #print shape(synth_art), self.m_used, art_traj
        synth_art[:, :] = art_traj.T
        #print "sound wave", self.synth.sound_wave(synth_art.T)
        return power * self.synth.sound_wave(synth_art.T)
