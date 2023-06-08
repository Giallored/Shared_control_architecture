#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np
from collections import deque

class Trajectory_smooter():
    def __init__(self, poly_degree=3, n_actions=20,rate=10):
        self.poly_degree = poly_degree
        self.n_actions = n_actions
        self.last_actions = deque([(0,0,0)]*n_actions,n_actions)
        self.rate=rospy.Rate(rate) # 10hz
        self.pub = rospy.Publisher('mobile_base_controller/ts_cmd_vel', Twist, queue_size=10)
        self.sub = rospy.Subscriber('mobile_base_controller/cmd_vel', Twist,self.callback)

    def main(self):
        rospy.init_node('trajectory_smoother', anonymous=True)
        while not rospy.is_shutdown():
            self.sub()

    def fit_polynomial(self):
        timesteps = np.array(range(0,self.n_actions))*1.0
        next_timestep = self.n_actions+1

        positions = np.array(list(self.last_actions)[:,0:1])
        l_poly = np.polyfit(timesteps, positions, self.poly_degree)
        linear_poly = np.poly2d(l_poly)

        angles = np.array(list(self.last_actions)[:,2])
        a_poly = np.polyfit(timesteps, angles, self.poly_degree)
        angular_poly = np.poly1d(a_poly)
        return list_to_twist([linear_poly[next_timestep]]+[angular_poly[next_timestep]])


    def callback(self,data):
        new_action = twist_to_list(data.data)
        self.last_actions.append(new_action)
        next_command = self.fit_polynomial()
        self.pub.publish(next_command)
        self.rate.sleep()
        rospy.loginfo(self.next_command)


def twist_to_list(twist_msg):
    return [twist_msg.linear.x,twist_msg.linear.y,twist_msg.angular.z]

def list_to_twist(l):
    vel_command = Twist()
    vel_command.linear.x = l[0]
    vel_command.linear.y = l[1]
    vel_command.angular.z = l[2]
    return vel_command


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
        trajectory_smooter()
        trajectory_smooter.main()
    except rospy.ROSInterruptException:
        pass