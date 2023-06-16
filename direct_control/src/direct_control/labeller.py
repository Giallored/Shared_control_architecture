#!/usr/bin/env python
import rospy
from direct_control.utils import *
from direct_control.dataset import *
from SC_navigation.environment import Environment

import numpy as np
from geometry_msgs.msg import Twist


class Labeller():
    def __init__(self, name_trial:str,max_size=1000000,rate=10):
        self.env = Environment()
        self.goal = self.env.goal_id
        self.buffer = Dataset(name=name_trial,max_size=max_size)
        self.prev_cmd=[0.,0.]
        self.pub=rospy.Publisher('mobile_base_controller/cmd_vel', Twist, queue_size=1)
        self.rate=rospy.Rate(rate) # 10hz        


    def main(self):
        print('Controller node is ready!')
        print('The GOAL of the simulation is: ',self.env.goal_id, ' in ', self.env.goal_pos)
        self.env.reset_sim()
        self.env.pause_sim()
        input('\n******** Press inv to start *********')
        countdown(3)
        self.env.unpause_sim()

        

        rospy.Subscriber('usr_cmd_vel',Twist,self.callback)
        rospy.spin()

    def callback(self,msg):
        self.env.update()
        cmd = twist_to_cmd(msg)
        is_full=self.buffer.is_full()
        if is_full==False:
            self.buffer.append(self.env.step,self.goal,self.env.cur_observation,cmd)            
        self.prev_cmd=cmd
        self.pub.publish(msg)
        self.rate.sleep()

    
