#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np

def talker():
    rospy.init_node('user', anonymous=True)
    pub = rospy.Publisher('usr_cmd_vel', Twist, queue_size=1)

    linear_cmds = [0.8,0.0,-0.8]
    angular_cmds = [0.5,0.0,-0.5]
    rate = rospy.Rate(10) # 10hz
    vel_cmd = Twist()
    while not rospy.is_shutdown():
        print('---------------------------')
        
        vel_cmd.linear.x= random.sample(linear_cmds ,1)[0]
        vel_cmd.angular.z=random.sample(angular_cmds,1)[0]
        rospy.loginfo(vel_cmd)
        pub.publish(vel_cmd)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass