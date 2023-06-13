#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import numpy as np
from geometry_msgs.msg import PoseArray,Pose
from std_msgs.msg import Bool,Float64MultiArray
from std_srvs.srv import Empty
#from arbitration.utils import list_to_twist
from utils.utils import list_to_twist


#from auto_controller.utils import list_to_twist
 

class Arbitrator():
    def __init__(self, alpha=0.6,rate=10):
        self.alpha=alpha
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_world', Empty) #resets the simulation
        self.steps=0
        self.max_steps = 100
        self.pub_cmd = rospy.Publisher('mobile_base_controller/cmd_vel', Twist, queue_size=1)
        self.pub=rospy.Publisher('request_cmd',Bool, queue_size=1)
        self.rate=rospy.Rate(rate) # 10hz


    def main(self):
        print('Arbitration node is ready!')
        self.reset_sim()
        rospy.Subscriber('autonomous_controllers/ts_cmd_vel', Float64MultiArray,self.callback)
        rospy.spin()
        
    
    def callback(self,data):
        self.steps +=1
        cmd = data.data
        n_agents = data.layout.dim[0].size
        v_cmd,om_cmd = blend_commands([1,1,1],cmd)
        msg = list_to_twist([v_cmd,0,0],[0,0,om_cmd])
        print('step_cnt = ',self.steps,' ---->  cmd [t =', rospy.get_time(),',] = ',[v_cmd,om_cmd])
        self.pub_cmd.publish(msg)
        self.pub.publish(Bool())
        if self.steps%self.max_steps ==0: self.reset_sim()
            

def blend_twist_commands(w_list,cmd_list):
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

def devide_cmd(cmd,dim):
    n = dim[0]
    l = dim[1]
    for i in range(n):
        cmd_i = cmd[n,n+1]



if __name__ == '__main__':
    try:
        rospy.init_node('arbitration_node', anonymous=True)
        node = Arbitrator()
        node.main()
    except rospy.ROSInterruptException:
        pass