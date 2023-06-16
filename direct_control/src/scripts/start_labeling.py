import argparse
from direct_control.utils import *
import rospy
from direct_control.labeller import Labeller

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Labelling process')

    parser.add_argument('--env',default='office',type=str, help='which is the world you are using')
    parser.add_argument('--folder', default='output_trajectories', type=str, help='')

    args = parser.parse_args()
    args.output = get_output_folder(args.folder, args.env)

    try:
        rospy.init_node('label_node', anonymous=True)
        node =Labeller(args.env)
        node.main()
    except rospy.ROSInterruptException:
        pass
    