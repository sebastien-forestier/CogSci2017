from explauto.environment.diva import DivaEnvironment
from .vrepEnvironment import VrepEnvironment
from explauto.environment.modular_environment import FlatEnvironment 


class VrepDivaEnvironment(FlatEnvironment):

    def __init__(self, environment, vrep, diva):
        
        FlatEnvironment.__init__(self, 
                                     environment, 
                                     [VrepEnvironment, 
                                     DivaEnvironment], 
                                     [vrep, 
                                     diva], 
                                     lambda l: l[0] + l[1])
