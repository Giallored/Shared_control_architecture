#!/usr/bin/env python

import rospy
from SC_navigation.navigation_controller import Controller

if __name__ == '__main__':
    try:
        rospy.init_node('navigation_controller', anonymous=True)
        node =Controller()
        node.main()
    except rospy.ROSInterruptException:
        pass