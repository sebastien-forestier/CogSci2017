import os
import time
import subprocess

import pypot.utils.pypot_time as ptime

from numpy import array, zeros, linspace

from pypot.primitive import LoopPrimitive
from pyvrep.xp import VrepXp
from explauto.utils import bounds_min_max, geomed
from explauto.environment.environment import Environment
from explauto.models.dmp import DmpPrimitive


class MovementPrimitive(LoopPrimitive):
    def __init__(self, environment, freq, motors, mov, primitive_duration, log):
        LoopPrimitive.__init__(self, environment.simulation.robot, freq*2.)
        self.motors = motors
        self.mov = mov
        self.primitive_duration = primitive_duration
        #print "mov", mov, mov.shape[0]
        self.one_step_duration = primitive_duration / mov.shape[0]
        self.log = log
        self.env = environment
        self.s_timepoint = 0
        #print "Movement primitive env. freq : ", freq, "mov size :", mov.shape[0], "duration : ", primitive_duration
        self.n_loops = 0
        
        
    def update(self):
        
        self.n_loops += 1
        
        if self.n_loops <= 2:
            self.t0 = ptime.time()
            #print "0elapsed time", self.elapsed_time
            
        #print "elapsed time", self.elapsed_time, "prim dur", self.primitive_duration, "one step", self.one_step_duration
        if self.elapsed_time > self.primitive_duration and self.s_timepoint == len(self.env.timepoints):
            self.stop(wait=False)
            return

        i = int(self.elapsed_time / self.one_step_duration)
        
        if i < 0:
            i = 0
            #print "Vrep time < 0", self.elapsed_time
            self.t0 = ptime.time()
            
        if i < self.mov.shape[0]:
            try:
                motor_positions = self.mov[i, :]
                self.env.set_motor_positions(motor_positions)
            except IndexError:
                print "INDEX ERROR"    
        #s = self.env.get_vrep_obj_position('head_visual')
        
        
        if self.s_timepoint < len(self.env.timepoints) and i >= self.env.timepoints[self.s_timepoint]-1:            
            s = self.env.get_sensor_state()
            s['i'] = self.env.timepoints[self.s_timepoint]
            #s['elapsed_time'] = self.elapsed_time
            self.env.s.append(s)
            self.s_timepoint += 1
            
            
        #if self.log:
            #self.env.emit('motor', motor_positions)
            #self.env.emit('sensori', s)
            
        
        


class VrepSimulation(VrepXp):
    def __init__(self, poppy_config, scene, gui):
        VrepXp.__init__(self, poppy_config, scene, gui=gui)
        
    def start(self):
        self._running.set()
        self.setup()
        
    def stop(self):
        self.teardown()
        
    
class VrepEnvironment(Environment):
    def __init__(self, **kwargs):
        for key, value in kwargs.iteritems():
            setattr(self, key, value)
            
        self.simulation = VrepSimulation(self.poppy_config, self.scene, self.gui)
        self.simulation.start()
        

        self.m_mins_deg = []
        self.m_maxs_deg = []
        
        self.uptime = 4.
        self.updates = 0
        
        time.sleep(8)
        self.init_motors()
        time.sleep(4)
        
        self.rest_position = self.deg2stds(self.rest_position_deg)
        self.reset_position()
        # angle_limits = []
        #print self.rest_position, len(rest_position)
        
        
        Environment.__init__(self, self.m_mins, self.m_maxs, self.s_mins, self.s_maxs)
        
        default = zeros(self.n_dmps_vrep*(self.n_bfs_vrep+2))
        if not self.vrep_use_initial:
            init_position = self.rest_position
            default[:self.n_dmps_vrep] = init_position
        if not self.vrep_use_goal:
            end_position = self.rest_position
            default[-self.n_dmps_vrep:] = end_position
        
        self.dmp = DmpPrimitive(self.n_dmps_vrep, 
                                self.n_bfs_vrep, 
                                self.used_vrep, 
                                default, 
                                type='discrete')
        
        

    def init_motors(self):
        
        self.motors_name = []
        for mot in sorted(self.simulation.robot.motors, key = lambda x: x.name):  # motors are not always in the same order
            #print mot.name
            self.motors_name.append(mot.name)
            if mot.name in self.constraints.keys():
                self.m_mins_deg.append(self.constraints[mot.name][0])
                self.m_maxs_deg.append(self.constraints[mot.name][1])
                #print mot.name, self.constraints[mot.name][0], self.constraints[mot.name][1]
            else:
                ll = mot.lower_limit
                ul = mot.upper_limit
                if ul > ll:
                    self.m_mins_deg.append(ll)
                    self.m_maxs_deg.append(ul)
                else:
                    self.m_mins_deg.append(ul)
                    self.m_maxs_deg.append(ll)
            
                    
                    
                
                
    def update(self, mov, log=False):
        #print "update vrep env"
        t = time.time()
        
        mov = self.compute_motor_command(mov)
        print "Primitive", mov
        #print "time_mov", time.time() - t
        #print "Movement goal : ", mov[-1]
        
        results = []
        for _ in range(self.vrepRepeat):
            self.reset()
            #head_init = self.get_vrep_obj_position('head_visual')
            
            
            
            self.s = []
            self.motor_primitive = MovementPrimitive(self, 
                                                     self.move_steps/self.move_duration, 
                                                     self.motors_name, 
                                                     mov, 
                                                     self.move_duration, 
                                                     log)
            
            #print "time_motor_primitive_start", time.time() - t
            #record video
            if self.recordvideo:
                
                FNULL = open(os.devnull, 'w')#doesnt work
                cmd = "timelimit -t3 -T10 recordmydesktop --no-sound -x 2680 -y 260 --width 800 --height 600 --fps 20 --v_quality 30 -o " + self.log_dir + "/video.ogv"
                print cmd
                self.recordprocess = subprocess.Popen(cmd.split(), stdout=FNULL)
            
            self.motor_primitive.start()
            self.motor_primitive.wait_to_stop()
            #print "m_env",len(m_env), m_env 
            s_ag = self.compute_sensori_effect(self.s)
            results.append(s_ag)
            


        self.uptime = self.uptime * (self.updates)/(self.updates+1) + (time.time() - t)/(self.updates+1)
        print "Update time", time.time() - t, "mean", self.uptime
        self.updates += 1
        
        #return array(array(self.s) - array(head_init))
        #print "vrepEnv updt", self.s, array(s_ag)
        return array(s_ag) if self.vrepRepeat == 1 else geomed(array(results))
    
    

    def get_vrep_obj_position(self, obj):
        io = self.simulation.robot._controllers[0].io
        #print "IO", obj, io.get_object_position(obj)
        return io.get_object_position(obj)

    def reset(self):
        #print "reset"
        #t = time.time()
        self.simulation.robot.reset_simulation()
        #print "time_reset", time.time() - t
        time.sleep(self.t_reset)
        self.reset_position()
        time.sleep(self.t_reset/4)
    
    def compute_motor_command(self, m_ag):
        return bounds_min_max(self.trajectory(m_ag), self.m_mins, self.m_maxs)

    def get_motor_names(self):
        return self.motors
    
    def get_motor_positions(self):
        positions = []
        for m in self.simulation.robot.motors:
            positions.append(m.present_position)
        return positions
        
        
    def std2deg(self, i_motor, motor_position):
        lb = self.m_mins_deg[i_motor]
        ub = self.m_maxs_deg[i_motor]
        return (motor_position + 1.) * ((ub - lb)/2.) + lb
        
    def deg2std(self, i_motor, motor_position_deg):
        lb = self.m_mins_deg[i_motor]
        ub = self.m_maxs_deg[i_motor]
        return (motor_position_deg - lb) / ((ub - lb)/2.) - 1.
    
    def std2degs(self, motor_positions):
        pos = []
        for i_motor, _ in enumerate(self.motors_name):
            pos.append(self.std2deg(i_motor, motor_positions[i_motor]))
        return pos
        
    def deg2stds(self, motor_positions_deg):
        pos = []
        for i_motor, _ in enumerate(self.motors_name):
            pos.append(self.deg2std(i_motor, motor_positions_deg[i_motor]))
        return pos
        
    def rest_params(self):
        dims = self.n_dmps_vrep*self.n_bfs_vrep
        if self.vrep_use_initial:
            dims += self.n_dmps_vrep
        if self.vrep_use_goal:
            dims += self.n_dmps_vrep
        rest = zeros(dims)
        if self.vrep_use_initial:
            rest[:self.n_dmps_vrep] = self.rest_position
        if self.vrep_use_goal:
            rest[-self.n_dmps_vrep:] = self.rest_position
        return list(rest)
    
    def reset_position(self):
        self.set_motor_positions(self.rest_position)
        
    def set_motor_position(self, motor_name, motor_position, scale = 'std'):
        ''' Suppose motor_position between -1 and 1 '''
        motor = getattr(self.simulation.robot, motor_name)
        if scale == 'std':      
            motor.goal_position = self.std2deg(self.motors_name.index(motor_name), 
                                               motor_position)
        elif scale == 'deg':
            motor.goal_position = motor_position
        
    def set_motor_positions(self, motor_positions, scale = 'std'):
        ''' Suppose motor_positions between -1 and 1 '''
        for i_motor, motor_name in enumerate(self.motors_name):
            self.set_motor_position(motor_name, 
                                    motor_positions[i_motor], 
                                    scale)
        
            
    def get_sensor_state(self):
        result = {}
        for vo in self.vrep_objects:             
            result[vo] = self.get_vrep_obj_position(vo)
        return result #dictionnary with objects positions


    def trajectory(self, m):
        y = self.dmp.trajectory(m)
        
        if len(y) > self.move_steps: 
            ls = linspace(0,len(y)-1,self.move_steps)
            ls = array(ls, dtype='int')
            y = y[ls]

        return y
        
        
    def compute_sensori_effect(self, s_mov):
        s = []
        #print s_mov
        for so in self.sensori_objects:
            tps = so['time_points']
            name = so['vrep_name']
            dims = so['dims']
            n = len(dims)
            #print tp, name, n, s_mov[tp][name]
            
            if so['dmp_sensor_bfs'] is not None:
                n_bfs_sensor = so['dmp_sensor_bfs']
                so['dmp_rest']
                        
                y = zeros((n,len(tps)))
                for si in s_mov:
                    if si['i'] in tps:
                        y[:,si['i']-1] = array(si[name])[dims]
                        
                        
                default_sensor = zeros(n*(n_bfs_sensor+2))
                default_sensor[:3] = so['dmp_rest']
                default_sensor[-3:] = so['dmp_rest']
                
                used_sensor = [False]*n + [True]*(n*n_bfs_sensor) + [False]*n
                
                dmp_sensor = DmpPrimitive(n, 
                                          n_bfs_sensor, 
                                          used_sensor, 
                                          default_sensor, 
                                          type='discrete')
                
                dmp_sensor.dmp.imitate_path(y)
                w = dmp_sensor.dmp.w.flatten()
                s += list(w)
                #print "DMP sensor y", y, "w", w
                
            else:
                for si in s_mov:
                    if si['i'] in tps:
                        s += list(array(si[name])[dims])
                        
    #                     except IndexError:
    #                         logging.warn('COULD NOT GET OBJECT POSITION :' + name)
    #                         print s_mov, len(s_mov)
    #                         s += list([0]*n)
        return s
