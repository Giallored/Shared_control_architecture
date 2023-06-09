#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np

def talker():
    pub = rospy.Publisher('mobile_base_controller/usr_cmd_vel', Twist, queue_size=10)
    rospy.init_node('user', anonymous=True)

    rate = rospy.Rate(5) # 10hz
    vel_command = Twist()
    while not rospy.is_shutdown():
        print('---------------------------')
        vel_command = get_rando_commands(vel_command)

        rospy.loginfo(vel_command)
        pub.publish(vel_command)
        rate.sleep()
#def get_position():
#    model_coordinates = rospy.ServiceProxy( '/gazebo/get_model_state', GetModelState)

def get_rando_commands(vel_command):
    vel_command.linear.x = random.random()
    vel_command.linear.y = random.random()
    vel_command.linear.z = 0#random.random()
    vel_command.angular.x = 0
    vel_command.angular.y = 0
    omega=np.random.normal(0, 5, 1)
    vel_command.angular.z = omega[0]
    return vel_command

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass