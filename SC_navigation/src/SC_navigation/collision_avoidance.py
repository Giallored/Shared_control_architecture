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


    def get_proj_point(self,sorted_cloud):
        cluster,n_points = self.get_cluster(sorted_cloud)
        centroid = np.array([cluster[:,0].sum(), cluster[:,1].sum()])/n_points
        rad = min([np.linalg.norm(centroid-p) for p in cluster])
        centroid_dist = np.linalg.norm(centroid)
        centroid_dir = -centroid/centroid_dist
        p_proj = np.array(centroid_dir)*(centroid_dist-rad)
        return p_proj


    def get_cmd(self,X_obs):
        theta = clamp_angle(np.arctan2(X_obs[1],X_obs[0]))   
        dist = np.linalg.norm(X_obs)

        sign = np.sign(theta)
        if sign==0.0: sign = 1

        if dist >=1.5:
            return [0.0,0.0],[0.0,0.0]
        
        

    
        #dU_r=self.d_Ur(X_obs,[0,0])   
        #F_v = np.array([dU_r[1],-dU_r[0]]) 
        #dx_d = F_v[0]
        #dy_d = F_v[1]
        #dtheta_d = np.arctan2(dy_d,dx_d)*(1.0+0.10*(theta/np.pi)**2)
        #dtheta_d = clamp_angle(dtheta_d)
        #v_cmd = self.K_lin*(theta/np.pi)**2
        #om_cmd = self.K_ang*(dtheta_d)* sign
        #return [v_cmd,om_cmd]
        
        #Rotational component
        dtheta_r = clamp_angle(theta + sign*np.pi/2)
        om_cmd = np.clip(-self.K_ang*(dtheta_r) ,self.min_om,self.max_om)
        #v_cmd = self.K_lin*(dtheta_r/np.pi)**4
        v_cmd = max(self.K_lin*(dtheta_r/np.pi)**4, self.k_r * (dist - self.dist_th)**2)
        
        v_cmd =  np.clip(v_cmd,self.min_v,self.max_v)
        cmd_r = [v_cmd,om_cmd]

        #translational component
        dtheta_t = clamp_angle(theta+np.pi)
        om_cmd = np.clip(self.K_ang*(dtheta_t),self.min_om,self.max_om)
        #v_cmd = self.K_lin*(dtheta_t/np.pi)**2
        v_cmd = max(self.K_lin*(dtheta_t/np.pi)**2, self.k_r * (dist - self.dist_th)**2)

        v_cmd = np.clip(v_cmd,self.min_v,self.max_v)
        cmd_t = [v_cmd,om_cmd]

        return cmd_r, cmd_t


    def get_cluster(self,sorted_cloud):
        p1 = sorted_cloud[0]
        p2 =sorted_cloud[1]
        p3 = sorted_cloud[2]
        d12 = np.linalg.norm(p1-p2)
        d13 = np.linalg.norm(p1-p3)
        d23 = np.linalg.norm(p2-p3)
        max_dist = max([d12,d13,d23])
        cluster =[p1,p2,p3]
        for p in sorted_cloud[3:]:
            dist_array = [np.linalg.norm(p_i-p) for p_i in cluster]
            if min(dist_array)<= max_dist:
                cluster.append(p)
                
            else:
                break
        return np.array(cluster),len(cluster)

    def d_Ur(self,X_p,X):
        x=X[0]
        y=X[1]
        x_p=X_p[0]
        y_p=X_p[1]
        ni = np.linalg.norm(X_p-X)
        #dU_x = (self.k_r/self.gamma) * (x-x_p) * (1/math.sqrt((x-x_p)**2+(y-y_p)**2) - 1/self.th_dist)**(self.gamma-1)
        #dU_y = (self.k_r/self.gamma) * (y-y_p) * (1/math.sqrt((x-x_p)**2+(y-y_p)**2) - 1/self.th_dist)**(self.gamma-1)
        dU_x = -self.gamma * self.k_r * (x_p-x) * (1/ni - 1/self.th_dist)**(self.gamma-1) / ni**3
        dU_y = -self.gamma * self.k_r * (y_p-y) * (1/ni - 1/self.th_dist)**(self.gamma-1) / ni**3

        return [dU_x,dU_y]