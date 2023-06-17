#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
from std_msgs.msg import Bool,String
from gazebo_msgs.msg import ModelStates
import numpy as np
from SC_navigation.utils import *



class User():
    
    def __init__(self,rate=10):
        self.tiago=TIAgo()
        self.pub = rospy.Publisher('usr_cmd_vel', Twist, queue_size=1)
        self.pub_goal = rospy.Publisher('goal', String, queue_size=1)
        self.goal_id='tiago'
        
        #self.pub = rospy.Publisher('usr_cmd_vel', Float64MultiArray, queue_size=1)
        
        #primitives
        self.stop_cmd = cmd_to_twist([0.0,0.0])
        self.up_cmd = cmd_to_twist([0.8,0.0])
        self.down_cmd = cmd_to_twist([-0.5,0.0])
        self.left_cmd = cmd_to_twist([0.0,-1.0])
        self.right_cmd = cmd_to_twist([0.0,1.0])
        

        #threshold on bearing to turn 
        self.theta_th = np.pi*0.01
       
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
        goal_rel_pos = self.tiago.get_relative_pos(self.goal_pos)

        if np.linalg.norm(goal_rel_pos)<0.01:
            #print('STOP')
            return self.stop_cmd 

        if goal_rel_pos[1]<0:
            theta = np.arctan2(goal_rel_pos[1],goal_rel_pos[0])
        else:
            theta = np.arctan2(goal_rel_pos[1],goal_rel_pos[0])

        if theta>np.pi:
            theta = theta- 2*np.pi

        #get the probability to get streight
        if abs(theta)>=self.theta_th:
            p=0.0
        else:
            p = self.theta_th/abs(theta)

        cmd = self.e_greedy_act(theta,p)
        
        return cmd
        

    def e_greedy_act(self,theta,p):
        epsilon= random.random()
        ##print(' - e: ',epsilon)
        if epsilon<=p: #straingth
            
            print('UP')
            return self.up_cmd
            #if abs(theta)>=3/4*np.pi:
            #    #print(' - | v |')
            #    return self.down_cmd
            #else:
            #    print(' - | ^ |')
            #    return self.up_cmd
        else:       #turn
            if theta>0:
                print('RIGHT')
                return self.right_cmd
            else:
                print('LEFT')
                return self.left_cmd
    
    def update(self):
        self.obj_dict = get_sim_info()
        self.goal_pos = Vec3_to_list(self.obj_dict[self.goal_id].position)
        self.tiago.set_MBpose(self.obj_dict['tiago'])  
     
    def choose_goal(self):
        obj_dict = get_sim_info()
        obs_ids = list(obj_dict.keys())
        obs_ids.remove('tiago')
        return random.sample(obs_ids,1)[0]
    
    def set_goal(self,data):
        self.goal_id = data.data
        #print(f"The new GOAL of the simulation is '{self.goal_id}'")

if __name__ == '__main__':
    try:
        rospy.init_node('User', anonymous=True)
        node =User()
        node.main()
    except rospy.ROSInterruptException:
        pass