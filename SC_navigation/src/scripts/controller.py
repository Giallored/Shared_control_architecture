#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np
from collections import deque
from std_msgs.msg import Bool
from SC_navigation.collision_avoidance import Collision_avoider
from SC_navigation.trajectory_smoother import Trajectory_smooter
import laser_geometry.laser_geometry as lg
from SC_navigation.environment import Environment
from SC_navigation.utils import twist_to_cmd,list_to_twist,cmd_to_twist,blend_commands
from RL_agent.DDPG import DDPG
from sensor_msgs.msg import PointCloud2,LaserScan




class Controller():
    
    def __init__(self,agent_args,training=True,rate=10,verbose=True):
        self.training=training
        self.ca_controller = Collision_avoider()
        self.ts_controller = Trajectory_smooter(dt=1.0/rate)
        self.agent = DDPG()
        self.env=Environment()
        self.prev_alpha=[1.0,1.0,1.0]
        self.pub=rospy.Publisher('mobile_base_controller/cmd_vel', Twist, queue_size=1)
        self.pub_request=rospy.Publisher('request_cmd',Bool,queue_size=1)
        self.rate=rospy.Rate(rate) # 10hz        
        self.verbose=verbose

        # Agent stuff
        self.nb_states=100
        self.nb_actions=3
        self.agent_args = agent_args



    def main(self):
        print('Controller node is ready!')
        self.env.reset_sim()
        rospy.Subscriber('scan_raw',LaserScan,self.update)
        rospy.Subscriber('usr_cmd_vel',Twist,self.callback)
        rospy.spin()



    def callback(self,data):
        self.env.n_step+=1

        #get commands
        usr_cmd = twist_to_cmd(data)
        ca_cmd=[0.,0.]# = self.ca_controller.get_cmd(self.env.obs)
        ts_cmd = self.ts_controller.get_cmd(self.env.time)

        alpha=self.get_arbitration()
        cmd=np.dot(alpha,[usr_cmd,ca_cmd,ts_cmd])
        
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)

        if self.verbose:
            self.display_commands()









    def display_commands(self,usr_cmd,ca_cmd,ts_cmd,cmd):
        print('STEP: ', self.env.n_step)
        print(' - user: ',usr_cmd)
        print(' - ca: ',ca_cmd)
        print(' - ts: ',ts_cmd)
        print('FINAL: ',cmd)
        print('---------------------')





        self.pub_request.publish(Bool())
        #print('step_cnt = ',self.steps,' ---->  cmd [t =', rospy.get_time(),',] = ',[v_cmd,om_cmd])
        time=rospy.get_time()
        self.ts_controller.store_action(time,cmd)
        self.rate.sleep()

    def update(self,scan_msg):
        time=rospy.get_time()
        self.env.update(time,scan_msg)


    def get_arbitration(self):
        alpha=self.prev_alpha

        return alpha


if __name__ == '__main__':
    try:
        rospy.init_node('navigation_controller', anonymous=True)
        node =Controller(training=False)
        node.main()
    except rospy.ROSInterruptException:
        pass