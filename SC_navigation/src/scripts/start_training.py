#!/usr/bin/env python

import rospy
from SC_navigation.navigation_controller import Controller
from RL_agent.utils import HyperParams
from SC_navigation.utils import get_output_folder
import os


if __name__ == '__main__':
    print('Lets go!')

    try:
        
        mode = rospy.get_param('/controller/mode') 
        rate = rospy.get_param('/controller/rate') 
        env_name = rospy.get_param('/controller/env') 
        verbose=rospy.get_param('/controller/verbose') 
        
        if mode=='train':
            is_training=True
        else:
            is_training=False
        train_param = HyperParams(rospy.get_param('/training'),is_training) 
        
        
        rospy.init_node('navigation_controller', anonymous=True,disable_signals=True)
        node =Controller(mode=mode,train_param=train_param,rate=rate,verbose=verbose)
        
        node.main()
        #rospy.init_node('navigation_controller', anonymous=True)
        #node =Controller()
        #node.main()
    except rospy.ROSInterruptException:
        pass