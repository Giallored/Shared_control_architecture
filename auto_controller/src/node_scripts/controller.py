#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np
from collections import deque
from std_msgs.msg import Bool
from auto_controller.collision_avoidance import Collision_avoider
from auto_controller.trajectory_smoother import Trajectory_smooter
from auto_controller.utils import blend_commands
import laser_geometry.laser_geometry as lg

from sensor_msgs.msg import PointCloud2,LaserScan




class Controller():
    
    def __init__(self,rate=10):
        self.ca_controller = Collision_avoider()
        self.ts_controller = Trajectory_smooter()
        self.lp = lg.LaserProjection()
        self.rate=rospy.Rate(rate) # 10hz
        self.pub_ca_cmd = rospy.Publisher('autonomous_controllers/ca_cmd_vel', Twist, queue_size=1)
        self.pub_ts_cmd = rospy.Publisher('autonomous_controllers/ts_cmd_vel', Twist, queue_size=1)
        #self.pub_scan_request = rospy.Publisher('autonomous_controllers/scan_request', Bool, queue_size=1)
        self.pub=rospy.Publisher('mobile_base_controller/cmd_vel', Twist, queue_size=1)
        

    def main(self):
        print('Controller node is ready!')
        rospy.Subscriber('aribitration/request_cmd',Bool,self.callback)
        rospy.spin()



    def callback(self,data):
        print("-----Request received at time: ",rospy.get_time(),'-------')

        # get user input
        #usr_cmd=rospy.wait_for_message("usr_cmd_vel", Twist, timeout=None)
        #print('usr_cmd = ',[usr_cmd.linear.x,usr_cmd.angular.z])

        # compute/publish the collision avoidance command
        scan = self.get_scan()
        ca_cmd = self.ca_controller.get_cmd(scan)
        self.pub_ca_cmd.publish(ca_cmd)
        print('ca_cmd = ',[ca_cmd.linear.x,ca_cmd.angular.z])

        # compute/publish the traj smoother command
        ts_cmd = self.ts_controller.get_cmd()
        self.ts_controller.previous_time = rospy.get_time()
        self.pub_ts_cmd.publish(ts_cmd)

        print('ts_cmd = ',[ts_cmd.linear.x,ts_cmd.angular.z])

        # Blend all inputs
        #vel_cmd = blend_commands([1,1],[usr_cmd,ca_cmd])
        ##print('FINAL_ts =',[vel_cmd.linear.x,vel_cmd.angular.z])
        #self.pub.publish(vel_cmd)
        #time=rospy.get_time()
        #self.ts_controller.store_action(time,vel_cmd)
        #print(self.ts_controller.last_actions)
        #print('-----------------------------------------------')
        self.rate.sleep()



    def get_scan(self):
        #self.pub_scan_request.publish(Bool())
        #scan = rospy.wait_for_message("autonomous_controllers/obs_pos",PointCloud2, timeout=None)
        scan_msg=rospy.wait_for_message("scan_raw",LaserScan, timeout=None)
        scan = self.lp.projectLaser(scan_msg)
        return scan


if __name__ == '__main__':
    try:
        rospy.init_node('auto_controller', anonymous=True)
        node =Controller()
        node.main()
    except rospy.ROSInterruptException:
        pass