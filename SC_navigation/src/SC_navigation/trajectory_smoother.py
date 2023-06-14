#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np
from collections import deque
from std_msgs.msg import Bool,Float64MultiArray
from auto_controller.utils import list_to_twist,twist_to_act,to_array_msg

class Trajectory_smooter():
    def __init__(self, dt,poly_degree=2, n_actions=3):
        self.poly_degree = poly_degree
        self.n_actions = n_actions
        self.previous_time=0
        self.last_actions = deque([],n_actions) #(time,v,omega)
        for i in range(self.n_actions): self.last_actions.append([i*dt,0,0])


    def get_cmd(self,time):
        dt = time - self.previous_time
        ts_cmd = self.fit_polynomial(dt)
        self.previous_time=time
        return ts_cmd

    def fit_polynomial(self,dt):
        actions = np.array(self.last_actions)
        timesteps = actions[:,0]
        next_time = timesteps[-1]+dt 
        
        v_cmds = actions[:,1]
        w_v = np.polyfit(timesteps, v_cmds, self.poly_degree)
        v_poly = np.poly1d(w_v)
        new_v_cmd = v_poly(next_time)

        om_cmds = actions[:,2]
        w_om = np.polyfit(timesteps, om_cmds, self.poly_degree)
        om_poly=np.poly1d(w_om)
        new_om_cmd = om_poly(next_time)

        return [new_v_cmd,new_om_cmd]

    def store_action(self,time,action):
        #time = rospy.get_time()
        #action = twist_to_act(vel_cmd)
        v = action[0]
        om = action[1]
        self.last_actions.append([time,v,om])

        #normalize actions
        min_time = self.last_actions[0][0]
        for i in range(self.n_actions): self.last_actions[i][0]=self.last_actions[i][0]-min_time
        print('User action stored : ',action)
