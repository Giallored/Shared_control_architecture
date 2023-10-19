#!/usr/bin/env python

from geometry_msgs.msg import Twist
import numpy as np
import matplotlib.pyplot as plt
from SC_navigation.utils import clamp_angle
import math
import os
import pickle



#from autonomous_controllers.utils import  FakeLaserScanner


class Collision_avoider():
    def __init__(self, delta=0.3,K_lin=1.0,K_ang=5.0,k_r=0.5,dir=None):
        self.dist_th=delta   #distance threshold
        self.K_lin= K_lin
        self.K_ang= K_ang
        self.k_r = k_r
        self.gamma = 2

        self.max_v = 0.8    
        self.min_v = -0.8
        self.max_om = 1.0
        self.min_om = -1.0


    def get_cmd(self,X_obs):
        theta = clamp_angle(np.arctan2(X_obs[1],X_obs[0]))   
        dist = np.linalg.norm(X_obs)

        sign = np.sign(theta)
        if sign==0.0: sign = 1

        if dist >=1.5:
            return [0.0,0.0],[0.0,0.0],[0.0,0.0]

        v_cmd = self.K_lin*max((theta/np.pi)**2,((dist - self.dist_th)/(1.5-0.3)**2))
        v_cmd = max((theta/np.pi)**2,((dist - self.dist_th)/(1.5-0.3)**2))
        v_cmd =  np.clip(v_cmd,self.min_v,self.max_v)
        
        #Rotational component
        dtheta_r = clamp_angle(theta + sign*np.pi/2)
        om_cmd = np.clip(-self.K_ang*(dtheta_r) ,self.min_om,self.max_om)
        cmd_r = [v_cmd,om_cmd]
        cmd_safety = [0,om_cmd]

        #translational component
        dtheta_t = clamp_angle(theta+np.pi)
        om_cmd = np.clip(self.K_ang*(dtheta_t),self.min_om,self.max_om)
        v_cmd = np.clip(v_cmd,self.min_v,self.max_v)
        cmd_t = [v_cmd,om_cmd]

        return cmd_r, cmd_t,cmd_safety