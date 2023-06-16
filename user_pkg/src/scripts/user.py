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
        #self.pub = rospy.Publisher('usr_cmd_vel', Float64MultiArray, queue_size=1)
        
        #primitives
        self.up_cmd = cmd_to_twist([0.8,0.0])
        self.down_cmd = cmd_to_twist([-0.5,0.0])
        self.left_cmd = cmd_to_twist([0.0,-1.0])
        self.right_cmd = cmd_to_twist([0.0,1.0])

        #threshold on bearing to turn 
        self.theta_th = np.pi*0.1
       
        #self.vel_cmd = Twist()
        self.rate=rospy.Rate(rate) # 10hz
        

    def main(self):
        print('User node is ready!')
        self.goal_id =self.choose_goal()

        print('GOAL: ',self.goal_id)
        while not rospy.is_shutdown():
            self.pub_goal.publish(String(self.goal_id))
            self.obj_dict = get_sim_info()
            self.goal_pos = Vec3_to_list(self.obj_dict[self.goal_id].position)
            self.tiago.set_MBpose(self.obj_dict['tiago'])  
            cmd = self.get_cmd()
            self.pub.publish(cmd)
            self.rate.sleep()

    def get_cmd(self):
        goal_rel_pos = self.tiago.get_relative_pos(self.goal_pos)

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

        #print('--------')
        #print(' - theta: ',theta)
        #print(' - p: ',p)
        cmd = self.e_greedy_act(theta,p)
        
        return cmd
        

    def e_greedy_act(self,theta,p):
        epsilon= random.random()
        #print(' - e: ',epsilon)
        if epsilon<=p: #straingth
            
            print(' - | ^ |')
            return self.up_cmd
            #if abs(theta)>=3/4*np.pi:
            #    print(' - | v |')
            #    return self.down_cmd
            #else:
            #    print(' - | ^ |')
            #    return self.up_cmd
        else:       #turn
            if theta>0:
                print(' - | > |')
                return self.right_cmd
            else:
                print(' - | < |')
                return self.left_cmd
            
     
    def choose_goal(self):
        obj_dict = get_sim_info()
        obs_ids = list(obj_dict.keys())
        obs_ids.remove('tiago')
        return random.sample(obs_ids,1)[0]

if __name__ == '__main__':
    try:
        rospy.init_node('User', anonymous=True)
        node =User()
        node.main()
    except rospy.ROSInterruptException:
        pass