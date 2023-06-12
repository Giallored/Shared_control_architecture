#!/usr/bin/python3
import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np
from geometry_msgs.msg import PoseArray,Pose
import math
from scipy.spatial.transform import Rotation

    
class FakeLaserScanner():
    def __init__(
        self, 
        angle_min=-1.9198600053787231,
        angle_max=1.9198600053787231,
        range_min=0.0,
        range_max=25.0
        ):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.range_min = range_min
        self.range_max = range_max

    def get_visible_obs(self,tiago,obstacles):
        visible_obs_pos = []
        visible_obs_id = []
        obs_ids = obstacles.ids
        for id in obs_ids:
            obs_i = obstacles.dict[id]
            is_visible,rel_pos_i = self.check_visibility(tiago,obs_i)
    
            if is_visible:
                visible_obs_pos.append(rel_pos_i)
                visible_obs_id.append(id)
            #if range_i>=self.range_min and range_i<=self.range_max:
            #    rel_pos_i = homoProd(tiago.Tf_tiago_w,pos_i)
            #    rel_pos.append(rel_pos_i)
        return visible_obs_pos,visible_obs_id
    
    def check_visibility(self,tiago,obs):
        tiago_pos = np.array(tiago.mb_position)
        pos_i= np.array(obs.position)
        tf = tiago.Tf_tiago_w
        rel_pos_i = homoProd(np.linalg.inv(tf),pos_i)
        range_i=np.linalg.norm(rel_pos_i)
        if range_i<self.range_min or range_i>self.range_max:
            return False,None
        bear_i = np.arctan2(rel_pos_i[1],rel_pos_i[0])
        if bear_i<self.angle_min or bear_i>self.angle_max:
            return False,None
        return True,list(rel_pos_i)

class TIAgo():
    def __init__(self, mb_position=[0.,0.,0.],mb_orientation=[0.,0.,0.]):
        self.mb_position=mb_position # wrt RFworld
        self.mb_orientation=mb_orientation # wrt RFworld
        self.Tf_tiago_w = np.zeros((4,4))
        #self.Tf_tiago_world

    def set_MBpose(self,new_pos,new_quat):
        self.mb_position=Vec3_to_list(new_pos)
        #turn orientation from quat to euler (in rad)
        rot = Rotation.from_quat(Vec4_to_list(new_quat))
        self.Tf_tiago_w = Pose2Homo(rot.as_matrix(),self.mb_position)
        self.mb_orientation=rot.as_euler('xyz', degrees=False)

    def get_MBpose(self):
        return self.mb_position+self.mb_position
    

class Obstacles:
    def __init__(self,obstacles_id, poses):
        self.ids=obstacles_id
        self.n_obs=len(self.ids)
        self.dict=dict()  #wrt RFworld
        self.set_poses(poses)

    def set_poses(self,poses):
        for i in range(self.n_obs):
            id_i=self.ids[i]
            pos_i=Vec3_to_list(poses[i].position)
            or_i = Vec4_to_list(poses[i].orientation)
            self.dict[id_i]= Object(id_i,pos_i,or_i)


class Object():
    def __init__(self,id, position, orientation):
        self.id=id
        self.position=position
        self.orientation=orientation
    
    def get_relative_pos(self,Tf):
        return Tf*self.position


def homoProd(Tf,vec3):
    vec4 = np.append(vec3,[1])
    prod = np.dot(Tf,vec4)
    return prod[:-2]
    

def Vec3_to_list(vector):
    return [vector.x,vector.y,vector.z]
def Vec4_to_list(vector):
    return [vector.x,vector.y,vector.z,vector.w]

def Pose2Homo(rot,trasl):
    p=np.append(trasl,[1])
    M=np.row_stack((rot,[0,0,0]))
    return np.column_stack((M,p))

def twist_to_list(twist_msg):
    return [twist_msg.linear.x,twist_msg.linear.y,twist_msg.angular.z]

def list_to_twist(l):
    vel_command = Twist()
    vel_command.linear.x = l[0]
    vel_command.linear.y = l[1]
    vel_command.angular.z = l[2]
    return vel_command






