#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import numpy as np
from geometry_msgs.msg import PoseArray,Pose
from std_msgs.msg import Bool,Float64MultiArray
from std_srvs.srv import Empty
#from arbitration.utils import list_to_twist
from utils.utils import list_to_twist,blend_commands


#from auto_controller.utils import list_to_twist
 

class Arbitrator():
    def __init__(self, alpha=0.6,rate=10):
        self.alpha=alpha
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_world', Empty) #resets the simulation
        self.steps=0
        self.max_steps = 100
        self.alpha=[1.0,1.0,1.0]
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

        v_cmd,om_cmd = blend_commands(self.alpha,cmd)
        msg = list_to_twist([v_cmd,0,0],[0,0,om_cmd])
        print('step_cnt = ',self.steps,' ---->  cmd [t =', rospy.get_time(),',] = ',[v_cmd,om_cmd])
        self.pub_cmd.publish(msg)
        self.pub.publish(Bool())
        if self.steps%self.max_steps ==0: self.reset_sim()
            




if __name__ == '__main__':
    try:
        rospy.init_node('arbitration_node', anonymous=True)
        node = Arbitrator()
        node.main()
    except rospy.ROSInterruptException:
        pass