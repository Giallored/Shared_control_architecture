import rospy
from geometry_msgs.msg import Twist
import numpy as np
from scipy.spatial.transform import Rotation
import time
from gazebo_msgs.msg import ModelStates


class TIAgo():
    def __init__(self, clear=0.2, mb_position=[0.,0.,0.],mb_orientation=[0.,0.,0.]):
        self.mb_position=mb_position # wrt RFworld
        self.mb_orientation=mb_orientation # wrt RFworld
        self.clear = clear
        self.Tf_tiago_w = np.zeros((4,4))

    def set_MBpose(self,pose):
        self.mb_position=Vec3_to_list(pose.position)
        #turn orientation from quat to euler (in rad)
        rot = Rotation.from_quat(Vec4_to_list(pose.orientation))
        self.Tf_tiago_w = Pose2Homo(rot.as_matrix(),self.mb_position)
        self.mb_orientation=rot.as_euler('xyz', degrees=False)

    def get_MBpose(self):
        return Vec3_to_list(self.mb_position),Vec3_to_listself.mb_position
    
    def get_relative_pos(self,obj_pos):
        pos = obj_pos+[1]
        rel_pos = np.dot(np.linalg.inv(self.Tf_tiago_w),pos)
        rel_pos=rel_pos[:-1]
        return rel_pos

def Pose2Homo(rot,trasl):
    p=np.append(trasl,[1])
    M=np.row_stack((rot,[0,0,0]))
    return np.column_stack((M,p))

def quat2euler(quat):
    rot = Rotation.from_quat(quat)
    return rot.as_euler('xyz', degrees=False)


def Vec3_to_list(vector):
    return [vector.x,vector.y,vector.z]
def Vec4_to_list(vector):
    return [vector.x,vector.y,vector.z,vector.w]

def twist_to_list(twist_msg):
    return [twist_msg.linear.x,twist_msg.linear.y,twist_msg.angular.z]

def list_to_twist(linear,angular):
    msg = Twist()
    msg.linear.x = linear[0]
    msg.linear.y = linear[1]
    msg.linear.z = linear[2]
    msg.angular.x = angular[0]
    msg.angular.y = angular[1]
    msg.angular.z = angular[2]
    return msg

def cmd_to_twist(cmd):
    msg = Twist()
    msg.linear.x = cmd[0]
    msg.angular.z = cmd[1]
    return msg

def twist_to_cmd(msg):
    return [msg.linear.x,msg.angular.z]

def blend_commands(w_list,cmd_list,n=3):
    #cmds = np.array_split(cmd_list, n)
    v=0
    om=0
    for i in range(n):
        w_i = w_list[i]
        v_i = cmd_list[i][0]
        v = v_i*w_i
        om_i = cmd_list[i][1]
        om+=w_i*om_i
    return v,om

def compute_cls_obs(obs_list):
        min_distace=999999999
        cls_point = [0,0]
        cls_id = 0
        cnt=0
        for obs in obs_list:
            distance = np.linalg.norm(obs)
            if distance<min_distace and distance > 0.0:
                min_distace=distance
                cls_obs=obs
                cls_id=cnt
            cnt+=1
        return cls_obs,min_distace
        
        
def from_cmd_msg(msg):
    cmds = np.array_split(msg.data, len(msg.data)/2)
    usr_cmd=cmds[0]
    ca_cmd = cmds[1]
    ts_cmd = cmds[2]
    return usr_cmd,ca_cmd,ts_cmd


def countdown(n):
    for i in (range(n,0,-1)): 
        print(f"{i}", end="\r", flush=True)
        time.sleep(1)
    print('GO!')



def get_sim_info():
    ms_msg = rospy.wait_for_message("/gazebo/model_states",ModelStates, timeout=None)
    ids = ms_msg.name
    poses = ms_msg.pose
    #tiago_i = ids.index('tiago')
    #ids.remove('tiago_i')
    #tiago_pose = poses.pop(tiago_i)
    dict = {}
    for id,pos in zip(ids,poses): dict[id]=pos
    
    return dict


def shout_down_routine(goal,reward):
    print("Mission accomplised")
    print(' - goal: ', goal)
    print(' - episode reward: ',reward)
    

