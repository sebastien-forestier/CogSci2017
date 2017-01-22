
# coding: utf-8

# In[5]:

import sys
sys.path.append('../')

from cogsci2017.environment.arm_diva_env import CogSci2017Environment
from cogsci2017.learning.supervisor import Supervisor
import matplotlib.pyplot as plt
import numpy as np



environment = CogSci2017Environment(gui=False, audio=False)


# In[6]:


plt.rc('text', usetex=True)
plt.rc('font', family='serif')



ax = plt.subplot()
fig = plt.gcf()
ax.set_aspect('equal')
fig.set_size_inches(6., 6., forward=True)

plt.xlabel("X", fontsize = 20)
plt.ylabel("Y", fontsize = 20)
plt.xlim([-2., 2.])
plt.ylim([-2., 2.])

plt.xticks([-2, -1, 0, 1, 2], ["$-1$", "$-0.5$", "$0$", "$0.5$", "$1$"], fontsize = 16)
plt.yticks([-2, -1, 0, 1, 2], ["$-1$", "$-0.5$", "$0$", "$0.5$", "$1$"], fontsize = 16)

environment.set_caregiver([0, 1.8])
plt.text(0.6, 1.7, "Caregiver", fontsize = 16)

environment.set_tool([-1., 0.2], 0.8)
plt.text(-1.45, -0.2, "Stick", fontsize = 16)

environment.set_toys([1.1, -1.3], [-1.1, 1.3], [0.9, -0.9])
plt.text(0.2, -1.35, "Toys", fontsize = 16)
plt.text(-0.9, 0.9, "Toy", fontsize = 16)

plt.text(0.4, 0., "Arm", fontsize = 16)

m = [-0.05]*21 + [0.]*28
environment.update(m)


environment.plot_step(ax, 10, clean=False)

plt.savefig('../figs/fig_env.pdf', format='pdf', bbox_inches='tight')


# In[ ]:



