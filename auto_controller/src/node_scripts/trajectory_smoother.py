#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np
from collections import deque
from std_msgs.msg import Bool,Float64MultiArray
from auto_controller.utils import list_to_twist,twist_to_act,to_array_msg

class Trajectory_smooter():
    def __init__(self, poly_degree=3, n_actions=20,rate=10):
        self.poly_degree = poly_degree
        self.n_actions = n_actions
        self.previous_time=0
        self.last_actions = deque([],n_actions) #(time,v,omega)
        for i in range(self.n_actions): self.last_actions.append([i*0.1,0,0])
        self.pub = rospy.Publisher('autonomous_controllers/ts_cmd_vel', Float64MultiArray, queue_size=10)
        self.rate=rospy.Rate(rate) # 10hz

    def main(self):
        print('TS node is ready!')
        rospy.Subscriber('autonomous_controllers/ca_cmd_vel', Float64MultiArray,self.callback)
        rospy.Subscriber('mobile_base_controller/cmd_vel', Twist,self.store_action)        
        rospy.spin()
    

    def callback(self,data):
        cmd = list(data.data)
        current_time = rospy.get_time()
        dt = current_time - self.previous_time
        ts_cmd = self.fit_polynomial(dt)
        print('cmd [t =', rospy.get_time(),'] = ',ts_cmd)
        print('-')
        cmd+=ts_cmd
        msg = to_array_msg(cmd,dim=[3,2])
        self.pub.publish(msg)
        self.previous_time = rospy.get_time()
        self.rate.sleep()

    def fit_polynomial(self,dt):
        actions = np.array(self.last_actions)
        timesteps = actions[:,0]
        next_time = timesteps[-1]+dt 

        v_cmds = actions[:,1]
        v_poly = np.polyfit(timesteps, v_cmds, self.poly_degree)
        new_v_cmd=float(np.poly1d(v_poly)[next_time])

        om_cmds = actions[:,2]
        om_poly = np.polyfit(timesteps, om_cmds, self.poly_degree)
        new_om_cmd = float(np.poly1d(om_poly)[next_time])
        list_to_twist([new_v_cmd,0,0],[0,0,new_om_cmd])

        return [new_v_cmd,new_om_cmd]

    def store_action(self,vel_cmd):
        time = rospy.get_time()
        action = twist_to_act(vel_cmd)
        v = action[0]
        om = action[1]
        self.last_actions.append((time,v,om))
        print('User action stored : ',action)





if __name__ == '__main__':
    try:
        rospy.init_node('trajectory_smoother', anonymous=True)
        node =Trajectory_smooter()
        node.main()
    except rospy.ROSInterruptException:
        pass