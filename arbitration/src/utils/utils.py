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

def blend_commands(w_list,cmd_list,n=3):
    cmds = np.array_split(cmd_list, n)
    v=0
    om=0
    for i in range(n):
        w_i = w_list[i]
        v_i = cmds[i][0]
        v = v_i*w_i
        om_i = cmds[i][1]
        om+=w_i*om_i
    return v,om