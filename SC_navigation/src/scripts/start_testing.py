#!/usr/bin/env python

import rospy
from SC_navigation.testing_controller import Controller
from RL_agent.utils import HyperParams
from SC_navigation.utils import get_output_folder
import os


if __name__ == '__main__':
    print('Lets go!')

    try:
        mode = rospy.get_param('/controller/mode') 
        rate = rospy.get_param('/controller/rate') 
        repeats = rospy.get_param('/controller/repeats') 
        verbose=rospy.get_param('/controller/verbose') 
        parent_dir = rospy.get_param('/training/parent_dir')
        shuffle = rospy.get_param('controller/shuffle')


        if mode == 'test':
            model = rospy.get_param('/controller/model') 
            weights_folder = rospy.get_param('/training/output')
            weights_dir = os.path.join(parent_dir,weights_folder)
            model_dir = os.path.join(weights_dir,model)
        else:
            model = 'classic'
            model_dir = ''

        test_dir = os.path.join(parent_dir,'testing')
        test_dir = os.path.join(test_dir, model)
        os.makedirs(test_dir, exist_ok=True)
        is_training = False
        train_param = HyperParams(rospy.get_param('/training'),is_training) 
        
        rospy.init_node('testing_controller', anonymous=True,disable_signals=True)
        node =Controller(mode=mode,
                         model_dir =model_dir,
                         test_dir = test_dir,
                         train_param=train_param,
                         repeats = repeats,
                         rate=rate,
                         shuffle=shuffle,
                         verbose=verbose)
        
        node.main()
    except rospy.ROSInterruptException:
        pass
