#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
from std_msgs.msg import Bool,Float64MultiArray,MultiArrayLayout
from auto_controller.utils import to_array_msg



class User():
    
    def __init__(self,rate=10):
        #self.pub_user_cmd = rospy.Publisher('usr_cmd_vel', Twist, queue_size=1)
        self.pub = rospy.Publisher('usr_cmd_vel', Float64MultiArray, queue_size=1)
        self.linear_cmds = [0.8,0.0,-0.8]
        self.angular_cmds = [0.5,0.0,-0.5]
        #self.vel_cmd = Twist()
        self.rate=rospy.Rate(rate) # 10hz
        

    def main(self):
        print('User node is ready!')
        rospy.Subscriber('request_cmd',Bool,self.callback)
        rospy.spin()

    def callback(self,data):
        cmd = []
        cmd.append(random.sample(self.linear_cmds ,1)[0])
        cmd.append(random.sample(self.angular_cmds,1)[0])
        print('cmd [t =', rospy.get_time(),'] = ',cmd)
        msg = to_array_msg(cmd,dim=[1,2])
        self.pub.publish(msg)
        #self.vel_cmd.linear.x= random.sample(self.linear_cmds ,1)[0]
        #self.vel_cmd.angular.z=random.sample(self.angular_cmds,1)[0]
        #self.pub_user_cmd.publish(self.vel_cmd)
        #self.rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('User', anonymous=True)
        node =User()
        node.main()
    except rospy.ROSInterruptException:
        pass