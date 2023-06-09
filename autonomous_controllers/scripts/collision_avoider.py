#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np
from geometry_msgs.msg import PoseArray,Pose
import math


class Collision_avoider():
    def __init__(self, delta=0.6,K_lin=1,K_ang=0.1,rate=10):
        self.delta=delta
        self.K_lin=K_lin
        self.K_ang=K_ang
        self.pub = rospy.Publisher('autonomous_controllers/ca_cmd_vel', Twist, queue_size=10)
        self.rate=rospy.Rate(rate) # 10hz

    def main(self):
        rospy.Subscriber("autonomous_controllers/obstacle_poses",PoseArray, self.callback)
        rospy.spin() # spin() simply keeps python from exiting until this node is stopped

    def callback(self,data):
        obs_poses=data.poses
        closest_obs_pos, closest_obs_distance = self.get_closest_obs(obs_poses)
        vel_cmd = self.compute_vel_cmd(closest_obs_pos,closest_obs_distance)
        self.pub.publish(vel_cmd)
        rospy.loginfo(vel_cmd)
        self.rate.sleep()
        #print('real command = ',vel_cmd)


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