#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
from std_msgs.msg import Bool,String
from gazebo_msgs.msg import ModelStates
import numpy as np
from SC_navigation.utils import *
from SC_navigation.robot_models import TIAgo




class User():
    
    def __init__(self,discrete = False,rate=5):
        rate = rospy.get_param('/controller/usr_rate') 
        self.tiago=TIAgo()
        self.pub = rospy.Publisher('usr_cmd_vel', Twist, queue_size=1)
        self.goal_id='tiago'
        self.discrete = discrete
        
        #self.pub = rospy.Publisher('usr_cmd_vel', Float64MultiArray, queue_size=1)
        
        #primitives
        self.primitive_v = 0.8
        self.primitive_om = 1.0
        self.stop_cmd = cmd_to_twist([0.0,0.0])


        self.max_v = 0.8
        self.min_v = -0.8
        self.max_om = 1.0
        self.min_om = -1.0
        self.k_l = 1
        self.k_a = 5        
        self.max_noise = 0.01
        

        #threshold on bearing to turn 
        self.theta_th = np.pi/3
       
        #self.vel_cmd = Twist()
        self.rate=rospy.Rate(rate) # 10hz
        
        model_msg = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=None)
        self.update(model_msg)

        

    def main(self):
        print('User node is ready!')
        rospy.Subscriber('goal',String,self.set_goal)
        rospy.Subscriber('gazebo/model_states',ModelStates,self.update)

        while not rospy.is_shutdown():
            cmd = self.get_cmd()
            self.pub.publish(cmd)
            self.rate.sleep()

    def get_cmd(self):
        dist,theta = self.get_goal_dist()
        #get the probability to turn

        if self.discrete:
            cmd = self.discrete_act(dist,theta)
        else:
            cmd = self.continue_act(dist,theta)

        return cmd


    

    def get_goal_dist(self):
        dist = np.subtract(self.goal_pos, self.tiago.mb_position)[0:2] #2d dist
        l_dist = np.linalg.norm(dist)
        a_dist = np.arctan2(dist[1],dist[0])
        r_theta =clamp_angle(self.tiago.mb_orientation[2])
        theta = a_dist-r_theta
        return l_dist,theta

    def discrete_act(self,dist,theta):
        if abs(theta)>self.theta_th:
            p=1.0
        else:
            p = (abs(theta)/self.theta_th)**2
        e= random.random()
        if dist<0.01:
            return self.stop_cmd 
        else:
            #get the linear command
            if abs(theta)<=self.theta_th:
                v = self.self.primitive_v
                om=0.0
            else:
                v = 0.0
            #get the angular command
            if e<=p:
                om = np.sign(theta)*self.primitive_om
            else:
                om=0.0
            cmd = cmd_to_twist([v,om])
            return cmd
        
    def continue_act(self,dist,theta):
        noise = np.random.uniform(-self.max_noise,self.max_noise,2)
        if abs(theta)<=self.theta_th:
            v = np.clip(self.k_l*dist+noise[0],self.min_v,self.max_v)
        else:
            v = 0.0
        om = np.clip(self.k_a*theta+noise[1],self.min_om,self.max_om)
        cmd = cmd_to_twist([v,om])
        return cmd



    
    def update(self,model_msg):
        self.obj_dict,vel = get_sim_info(model_msg)
        self.goal_pos = [6.0,0.0,0.01] #self.obj_dict[self.goal_id].position
        self.tiago.set_MBpose(self.obj_dict['tiago'],vel)  

    def set_goal(self,data):
        if data.data == 'END':
            rospy.signal_shutdown('Terminate training')
        else:
            self.goal_id = data.data
        #print(f"The new GOAL of the simulation is '{self.goal_id}'")

if __name__ == '__main__':
    try:
        rospy.init_node('User', anonymous=True)
        node =User(discrete = False)
        node.main()
    except rospy.ROSInterruptException:
        pass