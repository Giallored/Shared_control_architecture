#!/usr/bin/env python

import rospy
from arm_control.arm_controller import Controller
import os
import moveit_commander
import sys


if __name__ == '__main__':
    print('Lets go!') 

    #First initialize moveit_commander and a rospy node:
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("arm_controller", anonymous=True)
    node = Controller()
    node.main()
