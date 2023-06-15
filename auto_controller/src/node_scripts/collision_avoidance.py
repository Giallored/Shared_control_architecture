#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np
from geometry_msgs.msg import PoseArray,Pose
import math
from sensor_msgs.msg import PointCloud2,LaserScan
from auto_controller.point_cloud2 import read_points
from auto_controller.utils import to_array_msg
from std_msgs.msg import Bool,Float64MultiArray,MultiArrayLayout
import laser_geometry.laser_geometry as lg



#from autonomous_controllers.utils import  FakeLaserScanner


class Collision_avoider():
    def __init__(self, delta=0.6,K_lin=0.,K_ang=10.,rate=10):
        self.delta=delta
        self.K_lin=K_lin
        self.K_ang=K_ang
        self.lp = lg.LaserProjection()
        self.pub = rospy.Publisher('autonomous_controllers/ca_cmd_vel', Float64MultiArray, queue_size=10)

        #self.pub = rospy.Publisher('autonomous_controllers/ca_cmd_vel', Twist, queue_size=10)
        self.cnt_points=0
        self.rate=rospy.Rate(rate) # 10hz

    def main(self):
        print('CA node is ready!')
        #rospy.Subscriber('request_cmd',Bool,self.callback)
        rospy.Subscriber('usr_cmd_vel',Float64MultiArray,self.callback)
        rospy.spin()


    def callback(self,data):
        cmd = list(data.data)
        scan_msg = self.get_scan()
        cls_point = self.compute_cls_point(scan_msg)

        ca_cmd =self.compute_vel_cmd(cls_point) 
        print('cmd [t =', rospy.get_time(),'] = ',ca_cmd)
        cmd_new= cmd+ca_cmd
        msg = to_array_msg(cmd,dim=[2,2])
        self.pub.publish(msg)
        self.rate.sleep()

    def compute_cls_point(self,scan_msg):
        min_distace=999999999
        cls_point = [0,0]
        for p in read_points(scan_msg, skip_nans=True):
            point=[p[0], p[1]]#, data[2], data[3]]
            distance = np.linalg.norm(point)
            if distance<min_distace and distance > 0.0:
                min_distace=distance
                cls_point=point
            return cls_point

    
    def get_scan(self):
        scan_msg=rospy.wait_for_message("scan_raw",LaserScan, timeout=None)
        scan = self.lp.projectLaser(scan_msg)
        return scan


    def compute_vel_cmd(self,cls_point):
        obs_dir = np.subtract(cls_point,[0,0])
        repulsive_dir = -obs_dir/np.linalg.norm(obs_dir)
        repulsive_vel = self.K_lin*repulsive_dir
        repulsive_angle=np.arctan2(repulsive_dir[1],repulsive_dir[0])
        vel_cmd = [np.linalg.norm(repulsive_vel),self.K_ang*repulsive_angle]
        return vel_cmd



if __name__ == '__main__':
    try:
        rospy.init_node('CA_node', anonymous=True)
        node = Collision_avoider()
        node.main()
    except rospy.ROSInterruptException:
        pass