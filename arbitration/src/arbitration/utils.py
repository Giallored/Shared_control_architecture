import rospy
from geometry_msgs.msg import Twist



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

def blend_commands(w_list,cmd_list):
    cmd = Twist()
    for w,c in zip(w_list,cmd_list):
        cmd.linear.x +=w*c.linear.x
        cmd.linear.y +=w*c.linear.y
        cmd.linear.z +=w*c.linear.z
        cmd.angular.x+=w*c.angular.x
        cmd.angular.y+=w*c.angular.y
        cmd.angular.z+=w*c.angular.z
        #cmd.angular.w+=w*c.angular.w
    return cmd