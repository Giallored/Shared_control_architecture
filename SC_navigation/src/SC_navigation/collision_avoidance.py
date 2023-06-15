#!/usr/bin/env python

from geometry_msgs.msg import Twist
import numpy as np

from SC_navigation.utils import compute_cls_obs



#from autonomous_controllers.utils import  FakeLaserScanner


class Collision_avoider():
    def __init__(self, delta=0.7,K_lin=0.1,K_ang=0.1):
        self.delta=delta
        self.K_lin=K_lin
        self.K_ang=K_ang



    def get_cmd(self,obs_list):
        cls_obs,min_distance=compute_cls_obs(obs_list)
        if min_distance>self.delta:
            return [0.,0.]
        #print('Closest obstacle is: ',cls_obs, 'at ',min_distance)
        obs_dir = np.subtract(cls_obs,[0,0])
        repulsive_dir = -obs_dir/np.linalg.norm(obs_dir)

        if min_distance>0:
            lin_coeff =  self.K_lin/min_distance
            ang_coeff =  self.K_ang/min_distance
        else:
            lin_coeff=ang_coff = 100

        v_rep = -lin_coeff 
        repulsive_angle=ang_coeff*np.arctan2(repulsive_dir[1],repulsive_dir[0])
        om_rep = ang_coeff*repulsive_angle
        vel_cmd = [v_rep,om_rep]
        return vel_cmd

