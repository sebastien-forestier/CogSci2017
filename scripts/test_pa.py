import sys
import numpy as np
import time
from explauto.utils import rand_bounds
sys.path.append('../')

from src.environment.arm_diva_env import CogSci2017Environment


import pyaudio


env = CogSci2017Environment()


pa = pyaudio.PyAudio()

stream = pa.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=11025,
                            output=True)

for i in range(100):
    m = rand_bounds(env.conf.m_bounds)[0][21:]

    diva_traj = env.diva.update(m)

    sound = env.diva.sound_wave(env.diva.art_traj)

    print "sound", sound
    time.sleep(1)
    #stream.write(sound.astype(np.float32).tostring())