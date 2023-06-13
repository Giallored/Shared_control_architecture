#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import numpy as np
from geometry_msgs.msg import PoseArray,Pose


class Arbitrator():
    def __init__(self, alpha=0.6,rate=10):
        self.alpha=alpha
        self.pub = rospy.Publisher('mobile_base_controller/cmd_vel', Twist, queue_size=10)
        self.rate=rospy.Rate(rate) # 10hz


    def main(self):
        print('--------------------------')
        user_command = rospy.wait_for_message("autonomous_controllers/usr_cmd_vel", Twist, timeout=None)
        print('User_command!')
        ca_command = rospy.wait_for_message("autonomous_controllers/ca_cmd_vel", Twist, timeout=None)
        print('CA_command!')

        vel_cmd = blend_commands([self.alpha,1-self.alpha],[user_command,ca_command])
        self.pub.publish(vel_cmd)
        rospy.loginfo(vel_cmd)
        self.rate.sleep()
        rospy.spin()

def blend_commands(w_list,cmd_list):
    cmd = Twist()
    for w,c in zip(w_list,cmd_list):
        cmd.linear.x +=w*c.linear.x
        cmd.linear.y +=w*c.linear.y
        cmd.linear.z +=w*c.linear.z
        cmd.angular.x+=w*c.angular.x
        cmd.angular.y+=w*c.angular.y
        cmd.angular.z+=w*c.angular.z
        #cmd.angular.w+=w*c.angular.w
    return cmd


def Vec3_to_list(vector):
    return [vector.x,vector.y,vector.z]
def Vec4_to_list(vector):
    return [vector.x,vector.y,vector.z,vector.w]

def twist_to_list(twist_msg):
    return [twist_msg.linear.x,twist_msg.linear.y,twist_msg.angular.z]

def list_to_twist(l):
    vel_command = Twist()
    vel_command.linear.x = l[0]
    vel_command.linear.y = l[1]
    vel_command.angular.z = l[2]
    return vel_command


if __name__ == '__main__':
    try:
        rospy.init_node('arbitration_node', anonymous=True)
        node = Arbitrator()
        node.main()
    except rospy.ROSInterruptException:
        pass