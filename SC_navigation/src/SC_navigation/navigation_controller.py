#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import numpy as np
from collections import deque
from std_msgs.msg import Bool,String
from SC_navigation.collision_avoidance import Collision_avoider
from SC_navigation.trajectory_smoother import Trajectory_smooter
import laser_geometry.laser_geometry as lg
from SC_navigation.environment import Environment
from SC_navigation.utils import *
from RL_agent.ddpg import DDPG
from RL_agent.utils import HyperParams
from sensor_msgs.msg import PointCloud2,LaserScan
from copy import deepcopy



class Controller():
    
    def __init__(self,mode='test',train_param=None,rate=10,verbose=False):
       
        self.mode = mode
        self.env_name = rospy.get_param('/controller/env')   
        self.rate=rospy.Rate(rate) # 10hz        
        self.verbose=verbose
        self.load_model = rospy.get_param('/controller/model')

        #training stuff
        self.hyperParam = train_param
        self.prev_alpha=[1.0,1.0,1.0]
        self.prev_usr_cmd= [0.,0.]
        self.prev_cmd = [0.,0.]
        self.epoch=0
        self.episode_reward = 0.0
        self.mean_episode_rewards=[]
        self.mean_episode_losses=[]
        self.max_epochs = self.hyperParam.max_epochs
        

        #initializations
        self.ca_controller = Collision_avoider(
            delta=rospy.get_param('/controller/delta_coll'),
            K_lin=rospy.get_param('/controller/K_lin'),
            K_ang=rospy.get_param('/controller/K_ang'),
            )
        self.ts_controller = Trajectory_smooter()
        self.tiago = TIAgo(clear =rospy.get_param('/controller/taigoMBclear'))
        self.env = Environment(self.env_name,n_agents=3,
                               delta_coll= rospy.get_param('/controller/delta_coll'),
                               delta_goal= rospy.get_param('/training/delta_goal'),
                               max_steps=self.hyperParam.max_episode_length,
                               robot = self.tiago)

        self.agent = DDPG(self.env.n_states,self.env.n_actions,self.hyperParam)
        self.agent.reset(self.env.cur_observation)
        self.pub=rospy.Publisher('mobile_base_controller/cmd_vel', Twist, queue_size=1)
        self.pub_goal=rospy.Publisher('goal',String,queue_size=1)


        #directories

    


    def main(self):
        rospy.on_shutdown(self.shutdownhook)

        print('Controller node is ready!')
        print('The mode of the simulation is: ',self.mode)

        self.get_folder()
        self.reset()

        print(f'---- EPOCH {self.epoch} -------------------------')
        print(f"GOAL:'{self.env.goal_id}' -> {self.env.goal_pos}")

        #print('The GOAL of the simulation is: ',self.env.goal_id, ' in ', self.env.goal_pos)
        
        #self.env.pause_sim()
        #input('\n******** Press inv to start *********')
        #countdown(3)
        #self.env.unpause_sim()
        self.time=self.start_time

        if self.mode=='test':
            self.agent.load_weights(self.folder_dir)
            self.agent.is_training = False
            self.agent.eval() #makes the agents in evaluation mode
            self.decay_epsilon=False
            rospy.Subscriber('usr_cmd_vel',Twist,self.callback_SC)

        elif self.mode=='train':
            self.agent.is_training = True
            self.decay_epsilon=True
            rospy.Subscriber('usr_cmd_vel',Twist,self.callback_SC)
        elif self.mode == 'direct':
            rospy.Subscriber('usr_cmd_vel',Twist,self.callback_direct)
        else:
            rospy.ROSInterruptException            
        
        rospy.spin()

            
    def callback_direct(self,data):
        self.env.update()
        self.env.step += 1

        usr_cmd = twist_to_cmd(data)
        ca_cmd = self.ca_controller.get_cmd(self.env.obstacle_pos)
        ts_cmd = self.ts_controller.get_cmd(self.env.time)
        t=rospy.get_time()-self.start_time
        dt = round(t - self.time,3)
        self.time = t

        if ca_cmd==[0.0,0.0]:
            alpha=[1.,0.0,0.]
        else:
            alpha=[0.,1.0,0.]

        #print(f"STEP: {self.env.step} - Alpha = {[round(x,3) for x in alpha]}( (T) - dt = {dt}", end="\r", flush=True)
        cmd=np.dot(alpha,[usr_cmd,ca_cmd,ts_cmd])
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)

        # store the actions 
        self.ts_controller.store_action(self.time,cmd)
        self.env.update_act(alpha,usr_cmd,cmd)

        #store the plots
        self.plot.store(self.time,usr_cmd,ca_cmd,ts_cmd,alpha,cmd)
        self.plot.save_dict()
        if self.verbose:
            self.display_commands(alpha,usr_cmd,ca_cmd,ts_cmd,cmd)
        
        self.rate.sleep()




    def callback_SC(self,data):

        self.env.update()
        #get results fom the previous action and let the agent observe
        new_observation,reward,done = self.env.make_step()
        self.agent.observe(reward, new_observation, done)

        if self.mode == 'train' and (self.epoch>0 or self.env.step > self.hyperParam.warmup):
            # update policy only if the warmup is finished
            #self.env.pause_sim()
            self.agent.update_policy()
            #self.env.unpause_sim()

            # save and update
            if self.env.step % int(10) == 0:
                self.agent.save_model(self.folder_dir)

            
        #update
        self.env.step += 1
        self.episode_reward += reward
        self.observation = deepcopy(new_observation)


        if done: # end of episode
            self.epoch+=1
            mean_reward = self.episode_reward/self.env.step
            mean_val_loss = self.agent.episode_value_loss/self.env.step
            mean_policy_loss = self.agent.episode_policy_loss/self.env.step

            print('\n---')
            print(' - goal: ',self.env.goal_id)
            print(' - steps: ',self.env.step)
            print(' - epsilon: ',self.agent.epsilon)
            print(' - episode mean reward: ',mean_reward)
            print(' - episode mean val. loss: ',mean_val_loss)
            print(' - episode mean pol. loss: ',mean_policy_loss)
            print('---\n')
            self.plot.save_dict()
            if self.mode=='train' and self.epoch<self.max_epochs:
                self.mean_episode_rewards.append(mean_reward)
                self.mean_episode_losses.append([mean_val_loss,mean_policy_loss])
                print(f'---- EPOCH {self.epoch} -------------------------')
            else:
                self.env.pause()
                will = ""
                while not (will=='y' or will=='n'): 
                    will = str(input('Wanna save? Click "y" for yes or "n" for no:\n -> '))
                if will == 'y':
                    #self.plot.save_dict()
                    self.plot.save_plot()
                self.env.unpause()
            
            self.reset()
            print(f"GOAL:'{self.env.goal_id}' -> {self.env.goal_pos}")
            



        #NEW EPISODE
        #----------------------------------------------------------

        #get commands
        usr_cmd = twist_to_cmd(data)
        ca_cmd = self.ca_controller.get_cmd(self.env.obstacle_pos)
        ts_cmd = self.ts_controller.get_cmd(self.env.time)
        t=rospy.get_time()-self.start_time
        dt = round(t - self.time,3)
        self.time = t
        # get alpha from the DDPG    ==> as compute action
        if self.epoch==0 and self.mode == 'train' and self.env.step <= self.hyperParam.warmup:
            alpha = self.agent.random_action()
            print(f"STEP: {self.env.step} -> Alpha = {[round(x,3) for x in alpha]}(WU) - dt = {dt}", end="\r", flush=True)

        else:
            alpha = self.agent.select_action(self.observation,self.decay_epsilon)
            print(f"STEP: {self.env.step} - Alpha = {[round(x,3) for x in alpha]}( (T) - dt = {dt}", end="\r", flush=True)
        
         #alpha=[1.,0.,0.]

        # blend commands and send the msg to TIAgo 
        # (as the first part of step() in gym envs)
        cmd=np.dot(alpha,[usr_cmd,ca_cmd,ts_cmd])
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)


        #save all stuff for the second part of step()
        self.prev_alpha=alpha
        self.prev_observation=self.observation

        # store the actions 
        self.ts_controller.store_action(self.time,cmd)
        self.env.update_act(alpha,usr_cmd,cmd)

        #store the plots
        self.plot.store(self.time,usr_cmd,ca_cmd,ts_cmd,alpha,cmd)

        if self.verbose:
            self.display_commands(alpha,usr_cmd,ca_cmd,ts_cmd,cmd)
        
        self.rate.sleep()

    def get_folder(self):
        parent_dir = os.path.join(rospy.get_param('/training/parent_dir'),rospy.get_param('/training/output'))

        if self.mode =='train':
            if not self.load_model == "":
                self.folder_dir = os.path.join(parent_dir,self.load_model)
            else:
                self.folder_dir = get_output_folder(parent_dir,self.env_name)
        elif self.mode =='test':
            if self.load_model == "":
                self.load_model = input('What model to load? (check "weights" folder):\n -> ')
            self.folder_dir = os.path.join(parent_dir,self.load_model)
        elif self.mode =='direct':
            self.folder_dir = get_output_folder(parent_dir,'direct')
        else:
            self.folder_dir = None

        print('Directory: ',self.folder_dir)




    def display_commands(self,alpha,usr_cmd,ca_cmd,ts_cmd,cmd):
        print('STEP: ', self.env.step)
        print(' - alpha: ',alpha)
        print(' - user: ',usr_cmd)
        print(' - ca: ',ca_cmd)
        print(' - ts: ',ts_cmd)
        print('FINAL: ',cmd)
        print('---------------------')

    def get_arbitration(self):
        return self.prev_alpha




    def shutdownhook(self):
        print('Shout down...')

    def reset(self):
        print('RESET')
        self.env.reset()
        self.env.update()
        new_goal = self.env.goal_id
        self.pub_goal.publish(String(new_goal))
        self.episode_reward = 0.
        if not self.folder_dir==None:
            self.plot = Plot(
                goal=self.env.goal_id,
                env = self.env.name,
                parent_dir=self.folder_dir,
                name='epoch_'+str(self.epoch),
                description='')
            self.ca_controller.dir = self.plot.dir
        self.start_time = rospy.get_time()
        self.agent.reset(self.env.cur_observation)



