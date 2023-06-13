#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import numpy as np
from geometry_msgs.msg import PoseArray,Pose
from std_msgs.msg import Bool
#from arbitration.utils import *


class Arbitrator():
    def __init__(self, alpha=0.6,rate=10):
        self.alpha=alpha
        self.pub = rospy.Publisher('mobile_base_controller/cmd_vel', Twist, queue_size=10)
        self.pub_request=rospy.Publisher('aribitration/request_cmd',Bool, Twist, queue_size=10)
        self.rate=rospy.Rate(rate) # 10hz


    def main(self):
        while not rospy.is_shutdown():
            self.pub_request.publish(Bool())
            
            print('--------------------------')
            usr_cmd = rospy.wait_for_message("usr_cmd_vel", Twist, timeout=None)
            print(' - v_usr =',usr_cmd.linear.x)
            print(' - om_usr =',usr_cmd.angular.z)
            print('.')

            #print('User_cmd!')
            #print(usr_cmd)
            #ca_cmd = rospy.wait_for_message("autonomous_controllers/ca_cmd_vel", Twist, timeout=None)
            #print(' - v_ca =',ca_cmd.linear.x)
            #print(' - om_ca =',ca_cmd.angular.z)
            #print('.')

            #print('CA_cmd!')
            #print(ca_cmd)
            ts_cmd = rospy.wait_for_message("autonomous_controllers/ts_cmd_vel", Twist, timeout=None)
            print(' - v_ts =',ts_cmd.linear.x)
            print(' - om_ts =',ts_cmd.angular.z)
            print('.')

            vel_cmd = blend_commands([1,1],[usr_cmd,ts_cmd])
            print('FINAL_ts =',vel_cmd.linear.x)
            print('FINAL_ts =',vel_cmd.angular.z)
            #self.pub.publish(vel_cmd)
            #rospy.loginfo(vel_cmd)
            #self.rate.sleep()



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


if __name__ == '__main__':
    try:
        rospy.init_node('arbitration_node', anonymous=True)
        node = Arbitrator()
        node.main()
    except rospy.ROSInterruptException:
        pass