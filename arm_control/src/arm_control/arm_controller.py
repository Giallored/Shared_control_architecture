# Python 2/3 compatibility imports
from __future__ import print_function
from six.moves import input

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import Pose
import random
from std_msgs.msg import String,Bool
from moveit_commander.conversions import pose_to_list,list_to_pose
import numpy as np
from std_srvs.srv import Empty
import time



class Controller():
    
    def __init__(self,rate=5,verbose=True):
        self.rate=rospy.Rate(rate) # 10hz        
        self.verbose=verbose
        self.k = 0.01
        
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        
        #Instantiate a RobotCommander object
        self.robot = moveit_commander.RobotCommander()

        #Instantiate a PlanningSceneInterface object (provides a remote interface)
        self.scene = moveit_commander.PlanningSceneInterface()

        #Instantiate a MoveGroupCommander object
        self.group_name = "arm_torso"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.target_pose = self.move_group.get_current_pose().pose
        #dist = 1
        #while dist>0.01:
        #    error = self.execute_cmd()
        #    dist = np.linalg.norm(error)
        #    #print('Dist: ',dist)



    def main(self):
        rospy.Subscriber('usr_cmd/pose', Pose, self.change_target_pose)
        self.reset_sim()
        print('Arm controller is ready!')

        while not rospy.is_shutdown():
            error = self.execute_cmd()

            cur_pose = pose_to_list(self.move_group.get_current_pose().pose)

            print(f'Error: {[round(x,2) for x in error]}',end='\r',flush=True)
            self.rate.sleep()


    def change_target_pose(self,pose):
        self.target_pose = pose


    def execute_cmd(self):
        cur_pose = pose_to_list(self.move_group.get_current_pose().pose)
        error = np.subtract(pose_to_list(self.target_pose),cur_pose)
        goal_pose = np.array(cur_pose) + error*self.k
        goal_pose = list_to_pose(goal_pose.tolist())

        self.move_group.set_pose_target(goal_pose)
        success = self.move_group.go(wait=True)
        self.move_group.clear_pose_targets()

        new_error = np.subtract(pose_to_list(self.target_pose),cur_pose)
        return new_error

