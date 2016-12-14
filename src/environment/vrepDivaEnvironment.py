from explauto.environment.diva import DivaEnvironment
from .vrepEnvironment import VrepEnvironment
from .combined_env import CombinedEnvironment 


class VrepDivaEnvironment(CombinedEnvironment):

    def __init__(self, environment, vrep, diva):
        
        CombinedEnvironment.__init__(self, 
                                     environment, 
                                     VrepEnvironment, 
                                     DivaEnvironment, 
                                     vrep, 
                                     diva, 
                                     lambda l: l[0] + l[1])
