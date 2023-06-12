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
        self.pub = rospy.Publisher('autonomous_controllers/ca_cmd_vel', Twist, queue_size=10)
        self.pub_request = rospy.Publisher('autonomous_controllers/scan_request', Bool, queue_size=1)
        self.queue_is_empty=True
        self.cnt_points=0
        self.rate=rospy.Rate(rate) # 10hz

    def main(self):
        while not rospy.is_shutdown():
            request = rospy.wait_for_message("autonomous_controllers/request_cmd",Bool, timeout=None)
            self.pub_request.publish(Bool())
            rospy.Subscriber("autonomous_controllers/obs_pos",PointCloud2, self.callback)

    def callback(self,data):
        #print('Scan received!')
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
        print('--->cls_point is: ',cls_point,'at index ',cls_id)
        obs_dir = np.subtract(cls_point,[0,0])
        repulsive_dir = -obs_dir/np.linalg.norm(obs_dir)
        repulsive_vel = self.K_lin*repulsive_dir
        repulsive_angle=np.arctan2(repulsive_dir[1],repulsive_dir[0])
        
        linear_vel_cmd = np.linalg.norm(repulsive_vel)
        angular_vel_cmd = self.K_ang*repulsive_angle
        vel_cmd = Twist()
        vel_cmd.linear.x=linear_vel_cmd
        vel_cmd.angular.z=angular_vel_cmd
        self.pub.publish(vel_cmd)
        self.queue_is_empty=False
        self.rate.sleep()


    def get_closest_obs(self,obs_poses):
        min_dist =99999
        best_obs_pos = []
        for pose in obs_poses:
            obs_pos_i = Vec3_to_list(pose.position)
            dist_i= np.linalg.norm(obs_pos_i)
            if dist_i<min_dist:
                min_dist=dist_i
                best_obs_pos=obs_pos_i
        return best_obs_pos,min_dist

    def compute_vel_cmd(self,obs_pos,obs_dist):
        cmd = Twist()
        if obs_dist<self.delta:
            #cmd.linear.x=self.K_lin*(1/obs_dist-1/self.delta)
            #cmd.linear.x = self.K_lin*(self.delta-obs_dist)
            theta=np.arctan2(obs_pos[1],obs_pos[0])
            cmd.angular.z = self.K_ang*theta
        #print('------------------')
        #print(obs_dist<self.delta)
        #print(theta)
        #print('command = ',cmd)
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
        rospy.init_node('CA_node', anonymous=True)
        node = Collision_avoider()
        node.main()
    except rospy.ROSInterruptException:
        pass