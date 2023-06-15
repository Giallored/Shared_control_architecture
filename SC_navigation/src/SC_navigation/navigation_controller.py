#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import random
import numpy as np
from collections import deque
from std_msgs.msg import Bool
from SC_navigation.collision_avoidance import Collision_avoider
from SC_navigation.trajectory_smoother import Trajectory_smooter
import laser_geometry.laser_geometry as lg
from SC_navigation.environment import Environment
from SC_navigation.utils import *
from RL_agent.ddpg import DDPG
from sensor_msgs.msg import PointCloud2,LaserScan
from copy import deepcopy




class Controller():
    
    def __init__(self,args,rate=10,verbose=False):
        self.args = args
        self.rate=rospy.Rate(rate) # 10hz        
        self.verbose=verbose

        #initializations
        self.ca_controller = Collision_avoider()
        self.ts_controller = Trajectory_smooter(dt=1.0/rate)
        self.env = Environment(max_steps=100)
        self.agent = DDPG(self.env.n_states,self.env.n_actions,args)
        self.pub=rospy.Publisher('mobile_base_controller/cmd_vel', Twist, queue_size=1)
        self.pub_request=rospy.Publisher('request_cmd',Bool,queue_size=1)

        #training stuff
        self.n_iter = args.train_iter
        self.prev_alpha=[1.0,1.0,1.0]
        self.prev_usr_cmd= [0.,0.]
        self.prev_cmd = [0.,0.]
        self.episode=0
        self.episode_reward = 0.



    def main(self):
        print('Controller node is ready!')
        print('The GOAL of the simulation is: ',self.env.goal_id, ' in ', self.env.goal_pos)
        self.env.reset_sim()
        self.env.pause_sim()
        input('\n******** Press inv to start *********')
        countdown(3)
        self.env.unpause_sim()

        if self.args.mode=='test':
            rospy.Subscriber('usr_cmd_vel',Twist,self.callback_test)
            #rospy.Subscriber('scan_raw',LaserScan,self.update_env)

        else:
            rospy.Subscriber('usr_cmd_vel',Twist,self.callback_train)
        rospy.spin()


    def callback_test(self,data):
        print('-')
        self.env.update()
        reward=self.env.get_rewards()


        self.env.step+=1

        #get commands
        usr_cmd = twist_to_cmd(data)
        ca_cmd=self.ca_controller.get_cmd(self.env.cur_observation)

        #ca_cmd=[0.,0.]
        ts_cmd = self.ts_controller.get_cmd(self.env.time)
        ts_cmd=[0.,0.]
        #get arbitration, blend commands and send the msg to TIAgo 
        alpha=self.get_arbitration()
        cmd=np.dot(alpha,[usr_cmd,ca_cmd,ts_cmd])
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)

        # store the action for the TS controller
        time=rospy.get_time()
        self.ts_controller.store_action(time,cmd)
        self.env.update_act(alpha,usr_cmd,cmd)

        if self.verbose:
            self.display_commands(usr_cmd,ca_cmd,ts_cmd,cmd)
        self.rate.sleep()


    def display_commands(self,usr_cmd,ca_cmd,ts_cmd,cmd):
        print('STEP: ', self.env.step)
        print(' - user: ',usr_cmd)
        print(' - ca: ',ca_cmd)
        print(' - ts: ',ts_cmd)
        print('FINAL: ',cmd)
        print('---------------------')

    def get_arbitration(self):
        alpha=self.prev_alpha

        return alpha


    def callback_training(self,data):
        
        #get results fom the previous action and let the agent observe
        new_observation,reward,done = self.env.get_response()
        self.agent.observe(reward, new_observation, done)

        # update policy only if the warmup is finished
        if self.env.step > self.args.warmup :
            self.agent.update_policy()

        # save and update
        if self.env.step % int(10) == 0:
            self.agent.save_model(self.args.output)
        
        #update
        self.env.step += 1
        self.episode_steps += 1
        self.episode_reward += reward
        self.observation = deepcopy(new_observation)



        #NEW EPISODE
        #----------------------------------------------------------
        #get commands
        usr_cmd = twist_to_cmd(data)
        ca_cmd=[0.,0.]# = self.ca_controller.get_cmd(self.env.obs)
        ts_cmd = self.ts_controller.get_cmd(self.env.time)

        # get alpha from the DDPG    ==> as compute action
        if self.env.step <= self.args.warmup:
            alpha = self.agent.random_action()
        else:
            alpha = self.agent.select_action(self.observation)

        # blend commands and send the msg to TIAgo 
        # (as the first part of step() in gym envs)
        cmd=np.dot(alpha,[usr_cmd,ca_cmd,ts_cmd])
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)

        #save all stuff for the second part of step()
        self.prev_alpha=alpha
        self.prev_observation=self.observation

        # store the actions 
        time=rospy.get_time()
        self.ts_controller.store_action(time,cmd)
        self.env.update_act(alpha,usr_cmd,cmd)

        if self.verbose:
            self.display_commands(usr_cmd,ca_cmd,ts_cmd,cmd)
        self.env.step+=1
        self.rate.sleep()



# --------------------------------------------------------------------------
'''
    def train(self,num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):
    def train(self,args.train_iter, agent, env, evaluate, 
            args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)
        self.agent.is_training = True
        step = episode = episode_steps = 0
        episode_reward = 0.
        observation = None
        while step < self.n_iter:
            # reset if it is the start of episode
            if observation is None:
                observation = deepcopy(env.reset())
                self.agent.reset(observation)

            # agent pick action ...
            if step <= args.warmup:
                action = self.agent.random_action()
            else:
                action = self.agent.select_action(observation)
            
            # env response with next_observation, reward, terminate_info
            observation2, reward, done, info = env.step(action)
            observation2 = deepcopy(observation2)
            if max_episode_length and episode_steps >= max_episode_length -1:
                done = True

            # agent observe and update policy
            self.agent.observe(reward, observation2, done)
            if step > args.warmup :
                self.agent.update_policy()
            
            # [optional] evaluate
            if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
                policy = lambda x: agent.select_action(x, decay_epsilon=False)
                validate_reward = evaluate(env, policy, debug=False, visualize=False)
                if debug: prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))

            # [optional] save intermideate model
            if step % int(n_iter/3) == 0:
                self.agent.save_model(output)

            # update 
            step += 1
            episode_steps += 1
            episode_reward += reward
            observation = deepcopy(observation2)

            if done: # end of episode
                if debug: prGreen('#{}: episode_reward:{} steps:{}'.format(episode,episode_reward,step))

                self.agent.memory.append(
                    observation,
                    self.agent.select_action(observation),
                    0., False
                )

                # reset
                observation = None
                episode_steps = 0
                episode_reward = 0.
                episode += 1
'''

