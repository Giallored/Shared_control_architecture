#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import numpy as np
from gazebo_msgs.msg import ContactsState,ModelStates
import sys
from SC_navigation.robot_models import TIAgo
import random


#from gazebo_msgs.msg._ContactState.ContactState import Contact
from collections import deque
from std_msgs.msg import Bool,String
from SC_navigation.collision_avoidance import Collision_avoider
from SC_navigation.trajectory_smoother import Trajectory_smooter
import laser_geometry.laser_geometry as lg
from SC_navigation.environment import Environment
from SC_navigation.utils import *
from RL_agent.ddpg import DDPG
from RL_agent.ddqn import DDQN

from RL_agent.utils import HyperParams
from sensor_msgs.msg import PointCloud2,LaserScan
from copy import deepcopy



class Controller():
    
    def __init__(self,mode='test',train_param=None,rate=10,verbose=True):

        self.mode = mode
        self.env_name = rospy.get_param('/controller/env')   
        self.rate=rospy.Rate(rate) # 10hz        
        self.verbose=verbose
        
        self.n_agents=2
        self.n_acts = 2

        # folders
        self.output_folder = rospy.get_param('/training/output')
        self.plot_folder = rospy.get_param('/training/plot_folder')
        self.model2load = rospy.get_param('/controller/model')
        self.model_dir = self.get_folder(self.output_folder)
        self.plot_dir = self.get_folder(self.plot_folder)

        #training stuff
        
        self.hyperParam = train_param
        self.prev_alpha=[1.0]
        self.prev_usr_cmd= [0.,0.]
        self.prev_cmd = [0.,0.]
        self.epoch=0
        self.episode_reward = 0.0
        self.mean_episode_rewards=[]
        self.mean_episode_losses=[]
        self.max_epochs = self.hyperParam.max_epochs
        self.random_warmup = True
        self.step_warmap= 0
        self.eval_levels = 1
        self.eval_lev = None
        self.to_observe = True
        

        #initializations
        self.n_state = self.n_agents*self.n_acts + 1 + 2  #usr_cmd + ca_cmd + prev_alpha + cur_vel
        self.cur_cmds=[0.0]*self.n_acts*self.n_agents

        self.ca_controller = Collision_avoider(
            delta=rospy.get_param('/controller/delta_coll'),
            K_lin=rospy.get_param('/controller/K_lin'),
            K_ang=rospy.get_param('/controller/K_ang'),
            k_r=rospy.get_param('/controller/k_ac'),
            )
        self.ts_controller = Trajectory_smooter()
        self.tiago = TIAgo(clear =rospy.get_param('/controller/taigoMBclear'))
        self.env = Environment(self.env_name,n_agents=self.n_agents,
                               rate=rate,
                               delta_coll= rospy.get_param('/controller/delta_coll'),
                               theta_coll = rospy.get_param('/controller/theta_coll'),
                               delta_goal= rospy.get_param('/training/delta_goal'),
                               max_steps=self.hyperParam.max_episode_length,
                               robot = self.tiago)
        
        #self.agent = DDPG(self.n_state,self.hyperParam.n_frame,1,self.hyperParam)
        self.agent = DDQN(self.n_state,self.hyperParam.n_frame,5,self.hyperParam)
        
        self.pub=rospy.Publisher('mobile_base_controller/cmd_vel', Twist, queue_size=1)
        self.pub_goal=rospy.Publisher('goal',String,queue_size=1)


        self.train_plot = Plot(
            goal=self.env.goal_id,
            env = self.env.name,
            parent_dir=self.plot_dir,
            type = 'train')
    


    def main(self):
        self.start_time = rospy.get_time()
        if not self.model2load =="":
            self.agent.load_weights(self.model_dir)
            self.random_warmup = False
            where = os.path.join(self.plot_dir,'train_dict.pkl')
            with open(where, 'rb') as handle:
                dict = pickle.load(handle)
            self.train_plot.load_dict(dict)
            self.epoch = self.train_plot.epoch[-1]
            self.agent.epsilon=0.5

        print('Controller node is ready!')
        print(' - Mode: ',self.mode)
        print(f' - Model: {self.model2load} (epoch {self.epoch})')
        print(' - Directory: ',self.model_dir)

        #-------Setup
        self.reset()
        rospy.Subscriber('robot_bumper',ContactsState,self.env.callback_collision)
        rospy.Subscriber('gazebo/model_states',ModelStates,self.env.callback_robot_state)
        rospy.Subscriber('scan_raw',LaserScan,self.env.callback_scan)
        rospy.Subscriber('usr_cmd_vel',Twist,self.callback)

        rospy.on_shutdown(self.shutdownhook)

        #print(f'\n---- EPOCH {self.epoch} -------------------------')
        #print(f"GOAL:'{self.env.goal_id}' -> {self.env.goal_pos}\n.\n.\n.\n.")
        self.time=self.start_time
        rospy.spin()

    

    def callback(self, usr_msg):
        usr_cmd =np.array(twist_to_cmd(usr_msg)) #direct control

        if self.mode=='direct':
            self.pub.publish(usr_msg)

        elif self.mode=='classic':
            self.control_classic(usr_cmd)

        elif self.mode=='train':
            self.agent.is_training = True
            self.control_train(usr_cmd)

        elif self.mode=='eval' or self.mode =='test':
            self.agent.is_training = False
            self.control_test(usr_cmd)

        else:
            rospy.ROSInterruptException
            print('Mode is NOT supported!')


    def control_classic(self,usr_cmd):
        dt = self.get_time()
        _,_,done = self.env.make_step()
        if done:
            self.reset()
            print('-'*20+'\n.\n.\n.\n.\n.')

        ca_cmd = np.array(self.ca_controller.get_cmd(self.env.cls_obstacle)) 
        primitives = {1:(1.0,'U'),2:(0.75,'CA'),3:(0.5,'CA'),4:(0.25,'CA'),5:(0.0,'CA')}
        danger,dist = self.env.danger()
        
        alpha,tag = primitives[danger]
        alpha = 1.0
        header = 'STEP ' + str(self.env.step) +' - ' + tag

        if not (self.env.is_goal or self.env.is_coll):
            write_console(header,alpha, 0 ,danger,'-',dt)

        cmd = alpha*usr_cmd + (1-alpha)*ca_cmd        
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)
        
        t=rospy.get_time()-self.start_time
        self.time = t



        self.rate.sleep()




    def control_train(self,usr_cmd):
        is_warmup = self.step_warmap <= self.hyperParam.warmup
        dt = self.get_time()

        self.env.pause()
        #assemble and observe the current state
        observation,reward,done = self.env.make_step()
        ca_cmd = np.array(self.ca_controller.get_cmd(self.env.cls_obstacle))
        state_vars = np.hstack([usr_cmd,ca_cmd,self.prev_alpha, self.env.robot.mb_v,self.env.robot.mb_om])
        state = [observation,state_vars]
        self.agent.observe(reward, state, done, save=self.to_observe)

        if not is_warmup:
            self.agent.update_policy()
        
        self.episode_reward += reward

        if done: self.done_routine()

        #get the new action
        if is_warmup:
            self.step_warmap+=1
            alpha,a_opt = self.agent.select_action(state)
            if self.random_warmup:
                alpha = self.agent.random_action()
                header = 'WARMUP (random) - '+ str(self.step_warmap)
            else:
                header = 'WARMUP (true) - ' + str(self.step_warmap) 
        else:
            alpha,a_opt = self.agent.select_action(state)
            tag='(' + self.mode + ')'
            header = 'STEP ' + str(self.env.step) +' - ' + tag

        danger,dist = self.env.danger()
        if danger==5: alpha =0.0
        
        if not (self.env.is_goal or self.env.is_coll):
            write_console(header,alpha,a_opt,danger,self.agent.lr_scheduler.get_last_lr(),dt)
        
        # blend commands and send the msg to the robot
        cmd= usr_cmd * alpha + ca_cmd * (1-alpha)
        msg = cmd_to_twist(cmd)

        self.env.unpause()
        self.pub.publish(msg)

        # store actions & plots
        self.env.update_act(alpha,usr_cmd,cmd)
        self.plot.store_act(self.time,usr_cmd,ca_cmd,[0,0],alpha,cmd)
        self.plot.obs_poses[self.env.step]=self.env.pointCloud
        self.plot.ranges[self.env.step]=observation

        self.rate.sleep()



    def control_test(self,usr_cmd):
        dt = self.get_time()

        #assemble and observe the current state
        observation,_,done = self.env.make_step()
        ca_cmd = np.array(self.ca_controller.get_cmd(self.env.cls_obstacle))
        state_vars = np.hstack([usr_cmd,ca_cmd,self.prev_alpha, self.env.robot.mb_v,self.env.robot.mb_om])
        state = [observation,state_vars]

        if done: self.done_routine()

        #get the new action
        alpha,a_opt = self.agent.select_action(state)
        tag='(' + self.mode + ')'
        header = 'STEP ' + str(self.env.step) +' - ' + tag
        danger,dist = self.env.danger()
        if danger==5: alpha =0.0

        if not (self.env.is_goal or self.env.is_coll):
            write_console(header,alpha,a_opt,danger,self.agent.lr_scheduler.get_last_lr(),dt)
        
        # blend commands and send the msg to the robot
        cmd= usr_cmd * alpha + ca_cmd * (1-alpha)
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)

        # store actions 
        self.env.update_act(alpha,usr_cmd,cmd)

        self.rate.sleep()



    def get_folder(self,name):
        parent_dir = os.path.join(rospy.get_param('/training/parent_dir'),name)
        if self.mode =='train':
            if not self.model2load == "":
                return os.path.join(parent_dir,self.model2load)
            else:
                return  get_output_folder(parent_dir,self.env_name)
        elif self.mode =='test':
            if self.model2load == "":
                self.model2load = input('What model to load? (check "weights" folder):\n -> ')
            return os.path.join(parent_dir,self.model2load)
        elif self.mode =='direct':
            return get_output_folder(parent_dir,'direct')
        else:
            return None
        

    def shutdownhook(self):
        print('Shout down...')

    def reset(self):
        self.env.reset(self.eval_lev)
        self.agent.reset(self.env.observation,np.array([0.0,0.0,0.0,0.0,1.0,0.0,0.0]),1.0)
        self.ts_controller.reset()
        self.pub_goal.publish(String(self.env.goal_id))
        self.episode_reward = 0.
        if not self.plot_dir==None and not self.mode=='eval':
            self.plot = Plot(
                goal=self.env.goal_id,
                env = self.env.name,
                parent_dir=self.plot_dir,
                type = 'act',
                name=str(self.epoch)+'_epoch')
            self.ca_controller.dir = self.plot.dir

        self.start_time = rospy.get_time()
    
    def get_time(self):
        t=rospy.get_time()-self.start_time
        dt = round(t - self.time,3)
        self.time = t
        return dt
    


    def done_routine(self):

        self.env.pause()
                  
        mean_loss = self.agent.episode_loss/self.env.step
        self.mean_episode_losses.append(mean_loss)
        self.report()
    
        if self.mode=='train':
            epoch = self.epoch
            self.epoch+=1
            is_warmup = self.step_warmap <= self.hyperParam.warmup

            if mean_loss>0.001: #not evaluating
                self.train_plot.store_train(self.epoch,self.episode_reward,mean_loss)
                self.train_plot.save_dict()
                if mean_loss<=min(self.train_plot.mean_loss):
                    self.agent.save_model(self.model_dir)

            if epoch>=self.max_epochs:     #END of TRAINING
                self.pub_goal.publish('END')
                rospy.signal_shutdown('Terminate training')
                tag = 'END'

            elif is_warmup:                     #still  WARMUP
                tag = '(warm) EPOCH '+str(self.epoch)

            else:
                self.plot.save_dict()
                self.agent.update_hp()

                if epoch%5==0:
                    self.mode='eval'
                    self.eval_lev=0
                    tag = 'EVALUATION 0'
                else:
                    tag = '(train) EPOCH '+str(self.epoch)


        elif self.mode=='eval':   
            if self.env.is_goal:
                level = str(self.eval_lev)
                dir = os.path.join(self.model_dir,'level_'+level)
                os.makedirs(dir, exist_ok=True)
                self.agent.save_model(dir)
            self.eval_lev+=1
            tag = 'EVALUATION '+str(self.eval_lev)
            
            if self.eval_lev > self.eval_levels:
                self.eval_lev=None
                self.mode='train'
                self.agent.is_training = True
                tag = '(train) EPOCH '+str(self.epoch)
            
        else:           #this is in general for mode = 'test'
            self.env.pause()
            will = ""
            while not (will=='y' or will=='n'): 
                will = str(input('Wanna save? Click "y" for yes or "n" for no:\n -> '))
            if will == 'y':
                self.plot.save_dict()
            tag = 'TEST'
            
        self.env.unpause()
        self.reset()
        print(f'\n------------- {tag} -------------------------')
        print(f"GOAL:'{self.env.goal_id}' -> {self.env.goal_pos}\n.\n.\n.\n.\n")
    


    
    def report(self):
        print('\n---')
        print(' - mode: ',self.mode)
        #print(' - goal: ',self.env.goal_id)
        print(' - steps: ',self.env.step)
        print(' - episode reward: ',self.episode_reward)

        if self.mode == 'train':
            print(' - epsilon: ',self.agent.epsilon)
            print(f' - mem. capacity: {self.agent.buffer.get_capacity()*100}%')
            print(' - mean loss: ',self.mean_episode_losses[-1])
        print('---\n')


