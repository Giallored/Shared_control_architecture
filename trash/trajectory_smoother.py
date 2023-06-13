#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np
from collections import deque
from std_msgs.msg import Bool
from auto_controller.utils import twist_to_act,list_to_twist

class Trajectory_smooter():
    def __init__(self, poly_degree=3, n_actions=20,rate=10):
        self.poly_degree = poly_degree
        self.n_actions = n_actions
        self.previous_time=0
        self.last_actions = deque([],n_actions) #(time,v,omega)
        for i in range(self.n_actions): self.last_actions.append([i*0.01,0,0])
        self.rate=rospy.Rate(rate) # 10hz


    def fit_polynomial(self,dt):
        actions = np.array(self.last_actions)
        timesteps = actions[:,0]
        next_time = timesteps[-1]+dt 

        v_cmds = actions[:,1]
        v_poly = np.polyfit(timesteps, v_cmds, self.poly_degree)
        new_v_cmd=float(np.poly1d(v_poly)[next_time])

        om_cmds = actions[:,2]
        om_poly = np.polyfit(timesteps, om_cmds, self.poly_degree)
        new_om_cmd = float(np.poly1d(om_poly)[next_time])
        return list_to_twist([new_v_cmd,0,0],[0,0,new_om_cmd])


    def get_cmd(self):
        current_time = rospy.get_time()
        dt = current_time - self.previous_time
        next_command = self.fit_polynomial(dt)
        return next_command
       

    def store_action(self,time,vel_cmd):
        action = twist_to_act(vel_cmd)
        v = action[0]
        om = action[1]
        self.last_actions.append((time,v,om))






def get_rando_commands(vel_command):
    vel_command.linear.x = random.random()*10
    vel_command.linear.y = random.random()*10
    vel_command.linear.z = 0#random.random()
    vel_command.angular.x = 0
    vel_command.angular.y = 0
    omega=np.random.normal(0, 5, 1)
    vel_command.angular.z = omega[0]
    return vel_command

if __name__ == '__main__':
    try:
        rospy.init_node('trajectory_smoother', anonymous=True)

        node =Trajectory_smooter()
        node.main()
    except rospy.ROSInterruptException:
        pass