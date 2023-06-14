import rospy
from geometry_msgs.msg import Twist
import numpy as np



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
        for obs in obs_list:
            distance = np.linalg.norm(obs)
            if distance<min_distace and distance > 0.0:
                min_distace=distance
                cls_obs=obs
            return cls_obs,min_distace
        
        
def from_cmd_msg(msg):
    cmds = np.array_split(msg.data, len(msg.data)/2)
    usr_cmd=cmds[0]
    ca_cmd = cmds[1]
    ts_cmd = cmds[2]
    return usr_cmd,ca_cmd,ts_cmd