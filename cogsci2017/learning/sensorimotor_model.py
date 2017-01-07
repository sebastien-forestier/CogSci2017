
import numpy as np
from numpy import array
from explauto.exceptions import ExplautoBootstrapError
from explauto.sensorimotor_model.non_parametric import NonParametric
from explauto.utils import bounds_min_max
from explauto.utils import rand_bounds


class DemonstrableNN(NonParametric):
    def __init__(self, conf, sigma_explo_ratio=0.1, fwd='LWLR', inv='L-BFGS-B', **learner_kwargs):
        self.demonstrated = []
        NonParametric.__init__(self, conf, sigma_explo_ratio, fwd, inv, **learner_kwargs)
        
    def save(self):
        return [[self.model.imodel.fmodel.dataset.get_x(i) for i in range(len(self.model.imodel.fmodel.dataset))],
                [self.model.imodel.fmodel.dataset.get_y(i) for i in range(len(self.model.imodel.fmodel.dataset))],
                self.bootstrapped_s]
    
    def forward(self, data, iteration):
        self.model.imodel.fmodel.dataset.add_xy_batch(data[0][:iteration], data[1][:iteration])
        self.t = len(self.model.imodel.fmodel.dataset)
        if len(data) > 2:
            self.bootstrapped_s = data[2]
        else:
            self.bootstrapped_s = True

    def infer(self, in_dims, out_dims, x):
        if self.t < max(self.model.imodel.fmodel.k, self.model.imodel.k):
            res = rand_bounds(np.array([self.m_mins,
                                        self.m_maxs]))[0]
            return res, None

        if in_dims == self.m_dims and out_dims == self.s_dims:  # forward
            return array(self.model.predict_effect(tuple(x)))

        elif in_dims == self.s_dims and out_dims == self.m_dims:  # inverse
            if not self.bootstrapped_s:
                # If only one distinct point has been observed in the sensory space, then we output a random motor command
                res = rand_bounds(np.array([self.m_mins,
                                             self.m_maxs]))[0]
                sp = array(self.model.predict_effect(tuple(res)))
                return res, sp
            else:
                self.mean_explore = array(self.model.infer_order(tuple(x)))
                if self.mode == 'explore':
                    r = self.mean_explore
                    r[self.sigma_expl > 0] = np.random.normal(r[self.sigma_expl > 0], self.sigma_expl[self.sigma_expl > 0])
                    res = bounds_min_max(r, self.m_mins, self.m_maxs)
                    sp = array(self.model.predict_effect(tuple(res)))
                    return res, sp
                else:  # exploit'
                    return array(self.model.infer_order(tuple(x)))

        else:
            raise NotImplementedError
    
    def update(self, m, s):
        self.model.add_xy(tuple(m), tuple(s))
        self.t += 1
        if not self.bootstrapped_s and self.t > 1:
            if not (list(s[2:]) == list(self.model.imodel.fmodel.dataset.get_y(0)[2:])):
                self.bootstrapped_s = True