#!/usr/bin/env python
import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
#from utils import FakeLaserScanner,Obstacles,TIAgo
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseArray,Pose
import numpy as np
from scipy.spatial.transform import Rotation


class Sensor():
    def __init__(self, poly_degree=3, n_actions=20,rate=10):
        #self.laser_scanner=Fake_LaserScanner()
        self.tiago=TIAgo()
        self.scanner=FakeLaserScanner()
        self.pub = rospy.Publisher('autonomous_controllers/obstacle_poses', PoseArray, queue_size=1)
        self.rate=rospy.Rate(rate) # 10hz



    def main(self):
        rospy.Subscriber("/gazebo/model_states",ModelStates, self.callback)
        rospy.spin() # spin() simply keeps python from exiting until this node is stopped

    def callback(self,data):
        tiago_pose,obstacles = get_msg_info(data)
        self.tiago.set_MBpose(tiago_pose.position,tiago_pose.orientation)
        visible_obs_pos,visible_obs_id = self.scanner.get_visible_obs(self.tiago,obstacles)
        obstacle_poses = PoseArray()
        #obstacle_poses.append(tiago_pose)
        obstacle_poses.poses=visible_obs_pos
        self.pub.publish(obstacle_poses)
        rospy.loginfo(obstacle_poses)
        self.rate.sleep()



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
                visible_obs_pos.append(list_to_pose(rel_pos_i,[0]*4))
                visible_obs_id.append(id)
            
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
    return prod[:-1]
    

def Vec3_to_list(vector):
    return [vector.x,vector.y,vector.z]
def Vec4_to_list(vector):
    return [vector.x,vector.y,vector.z,vector.w]

def list_to_pose(position,orientation):
    pose=Pose()
    pose.position.x = position[0]
    pose.position.y = position[1]
    pose.position.z = position[2]
    pose.orientation.x = orientation[0]
    pose.orientation.y = orientation[1]
    pose.orientation.z = orientation[2]
    pose.orientation.w = orientation[3]
    return pose

def Pose2Homo(rot,trasl):
    p=np.append(trasl,[1])
    M=np.row_stack((rot,[0,0,0]))
    return np.column_stack((M,p))

def get_msg_info(msg):
    tiago_pose = msg.pose[-1]
    obs_ids= msg.name[:-1]
    obs_poses = msg.pose[:-1]
    obstacles = Obstacles(obs_ids,obs_poses)
    return tiago_pose,obstacles





if __name__ == '__main__':
    try:
        rospy.init_node('sensor_node', anonymous=True)
        sensor = Sensor()
        sensor.main()
    except rospy.ROSInterruptException:
        pass