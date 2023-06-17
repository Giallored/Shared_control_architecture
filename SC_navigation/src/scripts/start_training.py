import argparse
from RL_agent.utils import *
import rospy
from SC_navigation.navigation_controller import Controller
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')

    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--env',default='office',type=str, help='which is the world you are using')
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate') #was rate
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)') #was prate
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size') #was bsize
    parser.add_argument('--rmsize', default=6000000, type=int, help='memory size') #was rmsize
    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu') 
    parser.add_argument('--validate_episodes', default=20, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--max_episode_length', default=500, type=int, help='')
    parser.add_argument('--validate_steps', default=2000, type=int, help='how many steps to perform a validate experiment')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    
    
    args = parser.parse_args()
    args.output = get_output_folder(args.output, args.env)
    
    
    try:
        if args.mode == 'train' or args.mode =='test' :
            try:
                #os.system('rosrun user_pkg user.py')

                args.mode='test'
                rospy.init_node('navigation_controller', anonymous=True)
                node =Controller(args)
                node.main()
            except rospy.ROSInterruptException:
                pass
        else:
            raise RuntimeError('undefined mode {}'.format(args.mode))

    except rospy.ROSInterruptException:
        pass
    