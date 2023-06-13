#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np
from geometry_msgs.msg import PoseArray,Pose
import math
from sensor_msgs.msg import PointCloud2
from auto_controller.point_cloud2 import read_points
from auto_controller.utils import FakeLaserScanner
from std_msgs.msg import Bool

#from autonomous_controllers.utils import  FakeLaserScanner


class Collision_avoider():
    def __init__(self, delta=0.6,K_lin=0.,K_ang=10.,rate=10):
        self.delta=delta
        self.K_lin=K_lin
        self.K_ang=K_ang
        self.queue_is_empty=True
        self.cnt_points=0
        self.rate=rospy.Rate(rate) # 10hz

    def get_cmd(self,data):
        min_distace=999999999
        cls_point = [0,0]
        cnt=0
        for data in read_points(data, skip_nans=True):
            cnt+=1
            point=[data[0], data[1]]#, data[2], data[3]]
            distance = np.linalg.norm(point)
            if distance<min_distace and distance > 0.0:
                min_distace=distance
                cls_point=point
                cls_id = cnt
        #print('--->cls_point is: ',cls_point,'at index ',cls_id)
        obs_dir = np.subtract(cls_point,[0,0])
        repulsive_dir = -obs_dir/np.linalg.norm(obs_dir)
        repulsive_vel = self.K_lin*repulsive_dir
        repulsive_angle=np.arctan2(repulsive_dir[1],repulsive_dir[0])
        
        linear_vel_cmd = np.linalg.norm(repulsive_vel)
        angular_vel_cmd = self.K_ang*repulsive_angle
        vel_cmd = Twist()
        vel_cmd.linear.x=linear_vel_cmd
        vel_cmd.angular.z=angular_vel_cmd
        return vel_cmd
        
