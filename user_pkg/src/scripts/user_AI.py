#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
from std_msgs.msg import Bool,String
from gazebo_msgs.msg import ModelStates
import numpy as np
from SC_navigation.utils import *



class User():
    
    def __init__(self,rate=5):
        rate = rospy.get_param('/controller/usr_rate') 
        self.tiago=TIAgo()
        self.pub = rospy.Publisher('usr_cmd_vel', Twist, queue_size=1)
        self.goal_id='tiago'
        
        #self.pub = rospy.Publisher('usr_cmd_vel', Float64MultiArray, queue_size=1)
        
        #primitives
        self.stop_cmd = cmd_to_twist([0.0,0.0])
        self.up_cmd = cmd_to_twist([0.8,0.0])
        self.down_cmd = cmd_to_twist([-0.5,0.0])
        self.left_cmd = cmd_to_twist([0.0,-1.0])
        self.right_cmd = cmd_to_twist([0.0,1.0])
        

        #threshold on bearing to turn 
        self.theta_th = np.pi*0.3
       
        #self.vel_cmd = Twist()
        self.rate=rospy.Rate(rate) # 10hz
        

    def main(self):
        print('User node is ready!')
        rospy.Subscriber('goal',String,self.set_goal)
        while not rospy.is_shutdown():
            self.update()
            cmd = self.get_cmd()
            self.pub.publish(cmd)
            self.rate.sleep()

    def get_cmd(self):
        #goal_rel_pos = self.tiago.get_relative_pos(self.goal_pos)
        #goal_dist = np.linalg.norm(goal_rel_pos)
        #theta = np.arctan2(goal_rel_pos[1],goal_rel_pos[0])
        #theta = clamp_angle(theta)
        #print('theta: ', theta)

        dist,theta = self.get_goal_dist()
        #get the probability to turn
        if abs(theta)>self.theta_th:
            p=1.0
        else:
            p = (abs(theta)/self.theta_th)**2
        cmd = self.e_greedy_act(dist,theta,p)
        return cmd
    

    def get_goal_dist(self):
        dist = np.subtract(self.goal_pos, self.tiago.mb_position)[0:2] #2d dist
        l_dist = np.linalg.norm(dist)
        a_dist = np.arctan2(dist[1],dist[0])
        r_theta =clamp_angle(self.tiago.mb_orientation[2])
        theta = a_dist-r_theta
        return l_dist,theta

    def e_greedy_act(self,dist,theta,p):
        e= random.random()
        if dist<0.01:
            return self.stop_cmd 
        else:
            #get the linear command
            if abs(theta)<=self.theta_th:
                v = 0.8
                om=0.0
            else:
                v = 0.0
            #get the angular command
            if e<=p:
                om = np.sign(theta)*1.0
                #if theta>0:
                #    om= 1.0
                #else:
                #    om= -1.0
            else:
                om=0.0
            cmd = cmd_to_twist([v,om])
            return cmd



    
    def update(self):
        self.obj_dict = get_sim_info()
        self.goal_pos = self.obj_dict[self.goal_id].position
        self.tiago.set_MBpose(self.obj_dict['tiago'])  

    def set_goal(self,data):
        if data.data == 'END':
            rospy.signal_shutdown('Terminate training')
        else:
            self.goal_id = data.data
        #print(f"The new GOAL of the simulation is '{self.goal_id}'")

if __name__ == '__main__':
    try:
        rospy.init_node('User', anonymous=True)
        node =User()
        node.main()
    except rospy.ROSInterruptException:
        pass