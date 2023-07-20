#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import numpy as np
from gazebo_msgs.msg import ContactsState,ModelStates
import sys
from SC_navigation.robot_models import TIAgo


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
        self.eval_levels = 3
        self.eval_lev = 0
        

        #initializations
        self.n_state = self.n_agents*self.n_acts + 1 + 2  #usr_cmd + ca_cmd + prev_alpha + cur_vel
        self.cur_cmds=[0.0]*self.n_acts*self.n_agents

        self.ca_controller = Collision_avoider(
            delta=rospy.get_param('/controller/delta_coll'),
            K_lin=rospy.get_param('/controller/K_lin'),
            K_ang=rospy.get_param('/controller/K_ang'),
            )
        self.ts_controller = Trajectory_smooter()
        self.tiago = TIAgo(clear =rospy.get_param('/controller/taigoMBclear'))
        self.env = Environment(self.env_name,n_agents=self.n_agents,
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
        

        if not self.model2load =="":
            self.agent.load_weights(self.model_dir)
            self.random_warmup = False

        print('Controller node is ready!')
        print(' - Mode: ',self.mode)
        print(' - Model: ', self.model2load)
        print(' - Directory: ',self.model_dir)

        #-------Setup
        self.reset()
        rospy.Subscriber('robot_bumper',ContactsState,self.env.callback_collision)
        rospy.Subscriber('gazebo/model_states',ModelStates,self.env.callback_robot_state)
        rospy.Subscriber('scan_raw',LaserScan,self.env.callback_scan)
        rospy.on_shutdown(self.shutdownhook)

        print(f'\n---- EPOCH {self.epoch} -------------------------')
        print(f"GOAL:'{self.env.goal_id}' -> {self.env.goal_pos}\n.\n.\n.\n.")
        self.time=self.start_time

        if self.mode=='test':
            #self.agent.load_weights(self.model_dir)
            self.agent.is_training = False
            self.agent.eval() #makes the agents in evaluation mode
            rospy.Subscriber('usr_cmd_vel',Twist,self.callback_SC)
        elif self.mode=='train':
            self.agent.is_training = True
            rospy.Subscriber('usr_cmd_vel',Twist,self.callback_SC)
        elif self.mode == 'direct':
            rospy.Subscriber('usr_cmd_vel',Twist,self.callback_direct)

        elif self.mode == 'classic':
            rospy.Subscriber('usr_cmd_vel',Twist,self.callback_classic)
        else:
            rospy.ROSInterruptException
        
        rospy.spin()

    

    def callback_direct(self,data):        
        usr_cmd = twist_to_cmd(data)
        ca_cmd = self.ca_controller.get_cmd(self.env.pointCloud)
        ts_cmd = self.ts_controller.get_cmd(self.env.time)
        t=rospy.get_time()-self.start_time
        dt = round(t - self.time,3)
        self.time = t

        if ca_cmd==[0.0,0.0]:
            alpha=[1.,0.0,0.]
        else:
            alpha=[0.,1.0,0.]

        cmd=np.dot(alpha,[usr_cmd,ca_cmd,ts_cmd])
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)

        # store the actions 
        self.ts_controller.store_action(self.time,cmd)
        self.env.update_act(alpha,usr_cmd,cmd)

        #store the plots
        self.plot.store_act(self.time,usr_cmd,ca_cmd,ts_cmd,alpha,cmd)
        self.plot.save_dict()
        if self.verbose:
            self.display_commands_SC(alpha,usr_cmd,ca_cmd,ts_cmd,cmd)


    def callback_classic(self,data):
        _,_,done = self.env.make_step()

        usr_cmd =np.array(twist_to_cmd(data)) #direct control
        ca_cmd = np.array(self.ca_controller.get_cmd(self.env.cls_obstacle)) #collision avoidance  

        if self.env.danger_level()==2:
            alpha = 0.0 #(np.linalg.norm(self.env.cls_obstacle)/self.env.delta_coll)**3
            print('cmd: ',ca_cmd)
            module = 'User'

        else:
            alpha = 1.0
            module = 'CA'

        cmd = alpha*usr_cmd + (1-alpha)*ca_cmd
        #cmd = [usr_cmd, alpha*usr_cmd[1] + (1-alpha)*ca_cmd[1]]
        
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)
        
        t=rospy.get_time()-self.start_time
        self.time = t
        if done:
            
            self.reset()


        if self.verbose:
            self.display_commands_classic(module,cmd)
        
        self.rate.sleep()




    def callback_SC(self,data):
        #get results fom the previous action and let the agent observe
        observation,reward,done = self.env.make_step()
        if self.mode == 'train' and (self.epoch>0 or self.env.step > self.hyperParam.warmup):
            self.agent.update_policy()

        self.episode_reward += reward

        if done: self.done_routine()

        #NEW EPISODE
        #----------------------------------------------------------
        
        #update time
        t=rospy.get_time()-self.start_time
        dt = round(t - self.time,3)
        self.time = t
        usr_cmd = np.array(twist_to_cmd(data))
        danger = self.env.danger_level()

        #get commands
        ca_cmd = np.array(self.ca_controller.get_cmd(self.env.cls_obstacle))
        #ts_cmd = self.ts_controller.get_cmd(self.env.time)
        state_vars = np.hstack([usr_cmd,ca_cmd,self.prev_alpha, self.env.robot.mb_v,self.env.robot.mb_om])

        #assemble and observe the last episode data
        state = [observation,state_vars]

        self.agent.observe(reward,state, done)

        is_warmup = self.step_warmap > self.hyperParam.warmup
        # get alpha from the DDPG    ==> as compute action
        if self.mode == 'train' and is_warmup:
            self.step_warmap+=1
            alpha,a_opt = self.agent.select_action(state)
            if self.random_warmup:
                alpha = self.agent.random_action()
                header = 'WARMUP (random) - '+ str(self.step_warmap)
            else:
                header = 'WARMUP (true) - ' + str(self.step_warmap) 
        else:
            #print('\n ACT')
            alpha,a_opt = self.agent.select_action(state)
            tag='(' + self.mode + ')'
            header = 'STEP ' + str(self.env.step) +' - ' + tag

        if not (self.env.is_goal or self.env.is_coll):
            write_console(header,alpha,a_opt,danger,dt)
        # blend commands and send the msg to the robot
        cmd= usr_cmd * alpha + ca_cmd * (1-alpha)
    
        msg = cmd_to_twist(cmd)
        
        self.pub.publish(msg)
    

        #save all stuff for the second part of the step
        self.prev_alpha=alpha
        self.prev_observation=observation

        # store actions & plots
        self.ts_controller.store_action(self.time,cmd)
        self.env.update_act(alpha,usr_cmd,cmd)
        self.plot.store_act(self.time,usr_cmd,ca_cmd,[0,0],alpha,cmd)
        self.plot.obs_poses[self.env.step]=self.env.pointCloud
        self.plot.ranges[self.env.step]=observation

        if self.verbose:
            self.display_commands_SC(alpha,usr_cmd,ca_cmd,[0,0],cmd)
        
     
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
        self.env.reset(self.mode)
        self.agent.reset(self.env.observation,np.array([0.0,0.0,0.0,0.0,1.0,0.0,0.0]),1.0)
        self.ts_controller.reset()
        self.pub_goal.publish(String(self.env.goal_id))
        self.episode_reward = 0.
        if not self.plot_dir==None and self.mode=='train':
            self.plot = Plot(
                goal=self.env.goal_id,
                env = self.env.name,
                parent_dir=self.plot_dir,
                type = 'act',
                name=str(self.epoch)+'_epoch')
            self.ca_controller.dir = self.plot.dir

        self.start_time = rospy.get_time()


    def done_routine(self):
        self.env.pause()
        mean_reward = self.episode_reward/self.env.step
        self.mean_episode_rewards.append(mean_reward)            
        if self.agent.name == 'ddqn':
            mean_loss = self.agent.episode_loss/self.env.step
            self.mean_episode_losses.append(mean_loss)
            self.report_DDQN()
        else:
            mean_val_loss = self.agent.episode_value_loss/self.env.step
            mean_policy_loss = self.agent.episode_policy_loss/self.env.step
            self.mean_episode_losses.append([mean_val_loss,mean_policy_loss])
            self.report_DDPG()

        self.train_plot.store_train(self.epoch,mean_reward,mean_loss)


        if self.mode=='train':
            self.agent.save_model(self.model_dir)
            if self.epoch>=self.max_epochs:     #END of TRAINING
                self.pub_goal.publish('END')
                rospy.signal_shutdown('Terminate training')
            else:
                self.plot.save_dict()
                self.train_plot.save_dict()
                self.agent.update_hp()
                tag = '(train) EPOCH '+str(self.epoch)

                is_warmup = self.step_warmap > self.hyperParam.warmup

                if self.epoch%5==0 and not is_warmup: 
                    self.mode='eval'
                    self.agent.is_training = False
                    tag = 'EVALUATION'
                else:
                    self.epoch+=1


        elif self.mode=='eval':   
            
            if mean_reward>0:
                dir = os.path.join(self.model_dir,'level_'+str(self.eval_lev))
                self.agent.save_model(dir)
                self.eval_lev+=1
            
            if self.eval_lev > self.eval_levels:
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
        print(f"GOAL:'{self.env.goal_id}' -> {self.env.goal_pos}\n.\n.\n.\n.")
    

    def report_DDPG(self):
        print('\n---')
        print(' - mode: ',self.mode)
        #print(' - goal: ',self.env.goal_id)
        print(' - steps: ',self.env.step)
        if self.mode == 'train':
            print(' - sigma: ',self.agent.sigma_w)
            print(f' - mem. capacity: {self.agent.memory.actions.get_capacity()}%')
            print(' - mean episode reward: ',self.mean_episode_rewards[-1])
            print(' - mean episode val. loss: ',self.mean_episode_losses[-1][0])
            print(' - mean episode pol. loss: ',self.mean_episode_losses[-1][1])
        print('---\n')

    
    def report_DDQN(self):
        print('\n---')
        print(' - mode: ',self.mode)
        #print(' - goal: ',self.env.goal_id)
        print(' - steps: ',self.env.step)
        print(' - mean episode reward: ',self.mean_episode_rewards[-1])

        if self.mode == 'train':
            print(' - epsilon: ',self.agent.epsilon)
            print(f' - mem. capacity: {self.agent.buffer.get_capacity()*100}%')
            print(' - mean episode pol. loss: ',self.mean_episode_losses[-1])
        print('---\n')


