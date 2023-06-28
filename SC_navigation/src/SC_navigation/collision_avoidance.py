#!/usr/bin/env python

from geometry_msgs.msg import Twist
import numpy as np
import matplotlib.pyplot as plt
from SC_navigation.utils import compute_cls_obs,clamp_angle
import math
import os
import pickle



#from autonomous_controllers.utils import  FakeLaserScanner


class Collision_avoider():
    def __init__(self, delta=0.7,K_lin=0.1,K_ang=0.1,dir=None):
        self.th_dist=delta   #distance threshold
        self.K_lin=K_lin
        self.K_ang=K_ang
        self.k_r = 1.0
        self.gamma = 2
        self.frames={}
        self.f_i = 0
        
        if not dir==None:
            self.dir=dir
            self.save_frames=True
        else:
            self.save_frames=False

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

    def get_proj_point(self,sorted_cloud):
        cluster,n_points = self.get_cluster(sorted_cloud)
        centroid = np.array([cluster[:,0].sum(), cluster[:,1].sum()])/n_points
        rad = min([np.linalg.norm(centroid-p) for p in cluster])
        centroid_dist = np.linalg.norm(centroid)
        centroid_dir = -centroid/centroid_dist
        p_proj = np.array(centroid_dir)*(centroid_dist-rad)
        return p_proj



    def get_cmd(self,poin_cloud):

        dist_list = [np.linalg.norm(p) for p in poin_cloud]
        sorted_cloud =poin_cloud[np.argsort(dist_list)]
        try:
            min_dist = np.linalg.norm(sorted_cloud[0])
            X_obs = self.get_proj_point(sorted_cloud)
        except:
            return [0.0,0.0]
        
        
        if X_obs[1]>0: sign=1      #obstalce in thr Rx ==> CCW
        else: sign=-1                #obstalce in thr Lx ==> CW
        dU_r=self.d_Ur(X_obs,[0,0])   
        F_v = np.array([dU_r[1],-dU_r[0]]) * sign
        dx_d = F_v[0]
        dy_d = F_v[1]
        dtheta_d = np.arctan2(dy_d,dx_d)
        dtheta_d = clamp_angle(dtheta_d)
        
        v_cmd = -self.K_lin*(dx_d+dy_d)
        om_cmd = -self.K_ang*(dtheta_d)
        return [v_cmd,om_cmd]
        
            
        
    
    def get_frame(self,poin_cloud,cluster,X_obs,centroid):
        self.f_i+=1
        self.frames[self.f_i]={'point_cloud':poin_cloud,
                                'cluster':cluster,
                                'X_obs':X_obs,
                                'centroid':centroid}
        where = os.path.join(self.dir,'frames.pkl')
        with open(where, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        


        
   
        

        '''
        #print('Closest obstacle is: ',cls_obs, 'at ',min_distance)
        obs_dir = np.subtract(cls_obs,[0,0])
        repulsive_dir = -obs_dir/np.linalg.norm(obs_dir)

        if min_distance>0:
            lin_coeff =  self.K_lin/min_distance
            ang_coeff =  self.K_ang/min_distance
        else:
            lin_coeff=ang_coff = 100

        v_rep = -lin_coeff 
        repulsive_angle=np.arctan2(repulsive_dir[1],repulsive_dir[0])
        om_rep = ang_coeff*repulsive_angle
        vel_cmd = [v_rep,om_rep]
        return vel_cmd
        '''

        '''
        X_obs,min_dist =compute_cls_obs(poin_cloud)
        
        if min_dist<=self.th_dist:
            
            if X_obs[1]>0: sign=1      #obstalce in thr Rx ==> CCW
            else: sign=-1                #obstalce in thr Lx ==> CW
            dU_r=self.d_Ur(X_obs,[0,0])   

            F_v = np.array([dU_r[1],-dU_r[0]]) * sign
            dx_d = F_v[0]
            dy_d = F_v[1]
            dtheta_d = np.arctan2(dy_d,dx_d)
            dtheta_d = dtheta_d%(2*np.pi)
            if dtheta_d>np.pi:
                dtheta_d = dtheta_d-2*np.pi
            if dtheta_d<-np.pi:
                dtheta_d = dtheta_d+2*np.pi


            v_cmd = -self.K_lin*(dx_d+dy_d)
            om_cmd = -self.K_ang*(dtheta_d)
            print('dtheta_d: ',dtheta_d )

            return [v_cmd,om_cmd]
        else:
            return [0.,0.]
        '''

    