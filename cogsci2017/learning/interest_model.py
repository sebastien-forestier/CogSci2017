
import numpy as np

from explauto.interest_model.random import RandomInterest
from explauto.interest_model.competences import competence_dist
from explauto.models.dataset import Dataset


class MiscRandomInterest(RandomInterest):
    """
    Add some features to the RandomInterest random babbling class.
    
    Allows to query the recent interest in the whole space,
    the recent competence on the babbled points in the whole space, 
    the competence around a given point based on a mean of the knns.   
    
    """
    def __init__(self, 
                 conf, 
                 expl_dims,
                 win_size,
                 competence_mode,
                 k,
                 progress_mode):
        
        RandomInterest.__init__(self, conf, expl_dims)
        
        self.win_size = win_size
        self.competence_mode = competence_mode
        self.dist_max = np.linalg.norm(self.bounds[0,:] - self.bounds[1,:])
        self.k = k
        self.progress_mode = progress_mode
        self.data_xc = Dataset(len(expl_dims), 0)
        self.data_sr = Dataset(len(expl_dims), 0)
        self.data_sp = Dataset(len(expl_dims), 0)
        self.current_competence_progress = 0.
        self.current_prediction_progress = 0.
        self.current_progress = 0.
        self.current_interest = 0.
        self.alpha = 0.5
        self.beta = 1. - self.alpha
              
        
    def save(self):
        return [self.data_xc.data, 
                self.data_sr.data]
        
    def forward(self, data, iteration, progress, interest):
        self.data_xc.add_xy_batch(data[0][0][:iteration], data[0][1][:iteration])
        self.data_sr.data[0] = self.data_sr.data[0] + data[1][0][:iteration]
        self.data_sr.size += len(data[1][0][:iteration])
        self.current_progress = progress
        self.current_interest = interest
    
    def competence_measure(self, sg, s, dist_max):
        return competence_dist(sg, s, dist_max=dist_max)
    
    def add_xc(self, x):
        self.data_xc.add_xy(x)
        
    def add_sr(self, x):
        self.data_sr.add_xy(x)
        
    def add_sp(self, x):
        self.data_sp.add_xy(x)
        
    def update_interest(self, cp, pp):
        self.current_competence_progress += (1. / self.win_size) * (cp - self.current_competence_progress)
        self.current_prediction_progress += (1. / self.win_size) * (pp - self.current_prediction_progress)
        self.current_progress = self.alpha * self.current_competence_progress + self.beta * self.current_prediction_progress
        self.current_interest = self.alpha * abs(self.current_competence_progress) + self.beta * abs(self.current_prediction_progress)

    def update(self, xy, ms, sp=None):
        if sp is not None:
            c = self.competence_measure(xy[self.expl_dims], ms[self.expl_dims], dist_max=self.dist_max)
            p = self.competence_measure(sp, ms[self.expl_dims], dist_max=self.dist_max)
            
            cp = self.new_competence_progress(xy[self.expl_dims], c)
            pp = self.new_prediction_progress(ms[self.expl_dims], p)
            
            self.update_interest(cp, pp)
            self.add_xc(xy[self.expl_dims])
            self.add_sr(ms[self.expl_dims])
            self.add_sp(sp)
    
    def n_points(self):
        return len(self.data_xc)
                
    def new_competence_progress(self, x, c):
        if self.n_points() > 0:
            idx_sg_NN = self.data_xc.nn_x(x, k=1)[1][0]
            sr_NN = self.data_sr.get_x(idx_sg_NN)
            c_old = self.competence_measure(x, sr_NN, self.dist_max)
            return c - c_old
        else:
            return 0.
        
    def new_prediction_progress(self, sr, p):
        if self.n_points() > 0:
            idx_sr_NN = self.data_sr.nn_x(sr, k=1)[1][0]
            sp_NN = self.data_sp.get_x(idx_sr_NN)
            p_old = self.competence_measure(sr, sp_NN, self.dist_max)
            return p - p_old
        else:
            return 0.
        
    def interest_pt(self, x):
        if self.n_points() > self.k:
            _, idxs = self.data_xc.nn_x(x, k=self.k)
            idxs = sorted(idxs)
            v = [self.data_xc.get_y(idx) for idx in idxs]
            n = len(v)
            comp_beg = np.mean(v[:int(float(n)/2.)])
            comp_end = np.mean(v[int(float(n)/2.):])
            return np.abs(comp_end - comp_beg)
        else:
            return 0.
        
    def progress(self): return self.current_progress
    
    def interest(self): return self.current_interest      

        

class ContextRandomInterest(MiscRandomInterest):
    def __init__(self, 
                 conf, 
                 expl_dims,
                 win_size,
                 competence_mode,
                 k,
                 progress_mode,
                 context_mode):
        
        self.context_mode = context_mode
        
        MiscRandomInterest.__init__(self,
                                     conf, 
                                     expl_dims,
                                     win_size,
                                     competence_mode,
                                     k,
                                     progress_mode)        

              
    def competence_measure(self, csg, cs, dist_max):
        s = cs[self.context_mode["context_n_dims"]:]
        sg = csg[self.context_mode["context_n_dims"]:]
        return competence_dist(s, sg, dist_max=dist_max)
        