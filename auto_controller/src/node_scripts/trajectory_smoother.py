#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np
from collections import deque
from std_msgs.msg import Bool

class Trajectory_smooter():
    def __init__(self, poly_degree=3, n_actions=20,rate=10):
        self.poly_degree = poly_degree
        self.n_actions = n_actions
        self.last_actions = deque([(0,0,0)]*n_actions,n_actions) #(time,v,omega)
        self.rate=rospy.Rate(rate) # 10hz
        self.pub = rospy.Publisher('autonomous_controllers/ts_cmd_vel', Twist, queue_size=1)
        self.sub = rospy.Subscriber('mobile_base_controller/cmd_vel', Twist,self.callback)

    def main(self):
        print(rospy.get_time())
        self.previous_time = rospy.get_time()

        while not rospy.is_shutdown():
            rospy.Subscriber('aribitration/request_cmd',Bool,self.callback)


    def fit_polynomial(self,dt):
        timesteps = self.last_actions[:,0]
        next_time = timesteps[-1]+dt 

        v_cmds = self.last_actions[:,1]
        v_poly = np.polyfit(timesteps, v_cmds, self.poly_degree)
        new_v_cmd=np.poly1d(v_poly)[next_time]

        om_cmds = self.last_actions[:,2]
        om_poly = np.polyfit(timesteps, om_cmds, self.poly_degree)
        new_om_cmd = np.poly1d(om_poly)

        return list_to_twist([new_v_cmd,0,0]+[0,0,new_om_cmd])


    def callback(self,data):
        print('Request!')
        current_time = rospy.get_time()
        dt = current_time - self.previous_time
        next_command = self.fit_polynomial(dt)
        self.pub.publish(next_command)
        print('Request satisfied!')
        self.rate.sleep()
        final_vel_cmd = rospy.wait_for_message('mobile_base_controller/cmd_vel',Twist,timeout=None)
        self.store_action(final_vel_cmd)

    def store_action(self,data):
        time = rospy.get_time()
        action = twist_to_act(data.data)
        v = action[0]
        om = action[1]
        self.last_actions.append((time,v,om))




def twist_to_act(twist_msg):
    return [twist_msg.linear.x,twist_msg.angular.z]

def list_to_twist(linear,angular):
    vel_command = Twist()
    vel_command.linear.x = linear[0]
    vel_command.linear.y = linear[1]
    vel_command.linear.z = linear[2]

    vel_command.angular.x = angular[0]
    vel_command.angular.y = angular[1]
    vel_command.angular.z = angular[2]
    return vel_command


def get_rando_commands(vel_command):
    vel_command.linear.x = random.random()*10
    vel_command.linear.y = random.random()*10
    vel_command.linear.z = 0#random.random()
    vel_command.angular.x = 0
    vel_command.angular.y = 0
    omega=np.random.normal(0, 5, 1)
    vel_command.angular.z = omega[0]
    return vel_command

if __name__ == '__main__':
    try:
        rospy.init_node('trajectory_smoother', anonymous=True)

        node =Trajectory_smooter()
        node.main()
    except rospy.ROSInterruptException:
        pass