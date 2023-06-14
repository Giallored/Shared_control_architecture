#!/usr/bin/env python

from geometry_msgs.msg import Twist
import numpy as np

from SC_navigation.utils import compute_cls_obs



#from autonomous_controllers.utils import  FakeLaserScanner


class Collision_avoider():
    def __init__(self, delta=0.6,K_lin=0.,K_ang=1.):
        self.delta=delta
        self.K_lin=K_lin
        self.K_ang=K_ang



    def get_cmd(self,obs_list):
        cls_obs,min_distance=compute_cls_obs(obs_list)
        obs_dir = np.subtract(cls_obs,[0,0])
        repulsive_dir = -obs_dir/np.linalg.norm(obs_dir)
        repulsive_vel = self.K_lin*repulsive_dir
        repulsive_angle=np.arctan2(repulsive_dir[1],repulsive_dir[0])
        vel_cmd = [np.linalg.norm(repulsive_vel),self.K_ang*repulsive_angle]
        return vel_cmd

