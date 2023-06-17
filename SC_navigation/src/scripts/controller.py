#!/usr/bin/env python

import rospy
from SC_navigation.navigation_controller import Controller
from RL_agent.utils import get_output_folder,HyperParams

if __name__ == '__main__':
    print('Lets go!')



    try:
        
        mode = rospy.get_param('/controller/mode') 
        rate = rospy.get_param('/controller/rate') 
        env_name = rospy.get_param('/controller/env') 
        verbose=rospy.get_param('/controller/verbose') 
        if not mode == 'direct':
            train_param = HyperParams(rospy.get_param('/training')) 
            train_param.output= get_output_folder(train_param.output, env_name)
        else:
            train_param=None
        
        rospy.init_node('navigation_controller', anonymous=True)
        node =Controller(mode=mode,train_param=train_param,rate=rate,verbose=verbose)
        
        node.main()
        #rospy.init_node('navigation_controller', anonymous=True)
        #node =Controller()
        #node.main()
    except rospy.ROSInterruptException:
        pass