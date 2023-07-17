#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
import numpy as np
from gazebo_msgs.msg import ContactsState,ModelStates

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
        self.output_folder = rospy.get_param('/training/output')
        self.plot_folder = rospy.get_param('/training/plot_folder')
        self.model2load = rospy.get_param('/controller/model')
        self.n_agents=2
        self.n_acts = 2

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
        self.random_warmup = False
        

        #initializations
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
        
        self.n_state = self.n_agents*self.n_acts + 1 + 2  #usr_cmd + ca_cmd + prev_alpha + cur_vel
        
                    
        #self.agent = DDPG(self.n_state,self.hyperParam.n_frame,1,self.hyperParam)
        self.agent = DDQN(self.n_state,self.hyperParam.n_frame,11,self.hyperParam)
        
        self.pub=rospy.Publisher('mobile_base_controller/cmd_vel', Twist, queue_size=1)
        self.pub_goal=rospy.Publisher('goal',String,queue_size=1)

        self.cur_cmds=[0.0]*self.n_acts*self.n_agents
        #directories

    


    def main(self):
        self.model_dir = self.get_folder(self.output_folder)
        self.plot_dir = self.get_folder(self.plot_folder)

        if not self.model2load =="":
            self.agent.load_weights(self.model_dir)
            self.random_warmup = True

        print('Controller node is ready!')
        print(' - Mode: ',self.mode)
        print(' - Model: ', self.model2load)
        print(' - Directory: ',self.model_dir)

        #-------Setup
        self.reset()
        rospy.Subscriber('robot_bumper',ContactsState,self.env.callback_collision)
        rospy.Subscriber('gazebo/model_states',ModelStates,self.env.callback_robot_state)
        rospy.on_shutdown(self.shutdownhook)


        print(f'---- EPOCH {self.epoch} -------------------------')
        print(f"GOAL:'{self.env.goal_id}' -> {self.env.goal_pos}")
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
        self.env.update()

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
        self.plot.store(self.time,usr_cmd,ca_cmd,ts_cmd,alpha,cmd)
        self.plot.save_dict()
        if self.verbose:
            self.display_commands_SC(alpha,usr_cmd,ca_cmd,ts_cmd,cmd)
        
        self.rate.sleep()



    def callback_classic(self,data):

        self.env.update()
        _,_,done = self.env.make_step()

        is_safe,cls_obs_dist = self.env.safety_check()
        usr_cmd =np.array(twist_to_cmd(data)) #direct control
        ca_cmd = np.array(self.ca_controller.get_cmd(self.env.pointCloud)) #collision avoidance  

        if not is_safe:
            alpha = (1-(cls_obs_dist/self.env.delta_coll)**2)
            module = 'User'

        else:
            alpha = 1.0
            module = 'CA'

        print('usr_cmd: ',usr_cmd)
        print('ca_cmd: ',ca_cmd)
        print('alpha: ',alpha)
        cmd = alpha*usr_cmd + (1-alpha)*ca_cmd
            
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
        self.env.update()

        #get results fom the previous action and let the agent observe
        observation,reward,done = self.env.make_step()
        #print('reward: ',reward)
        
        if self.mode == 'train' and (self.epoch>0 or self.env.step > self.hyperParam.warmup):
            #print('\n TRAIN')
            # update policy only if the warmup is finished
            self.agent.update_policy()

        
        #update parameters
        
        self.episode_reward += reward


        if done: # end of episode
            self.env.pause()
            self.epoch+=1
            mean_reward = self.episode_reward/self.env.step
            self.mean_episode_rewards.append(mean_reward)            

            if self.agent.name == 'ddqn':
                self.mean_episode_losses.append(self.agent.episode_loss/self.env.step)
                self.report_DDQN()
            else:
                mean_val_loss = self.agent.episode_value_loss/self.env.step
                mean_policy_loss = self.agent.episode_policy_loss/self.env.step
                self.mean_episode_losses.append([mean_val_loss,mean_policy_loss])
                self.report_DDPG()

            if self.mode=='train':
               
                #save and update
                #if self.env.step % int(10) == 0:
                self.agent.save_model(self.model_dir)
                if self.epoch>=self.max_epochs:     #END of TRAINING
                    self.pub_goal.publish('END')
                    rospy.signal_shutdown('Terminate training')
                else:
                    self.plot.save_dict()

                    self.agent.update_hp()

                    if self.epoch%5==0: #every n steps you do an evaluation run
                        self.mode='eval'
                        self.agent.is_training = False
                        print(f'---- EVALUATION -------------------------')
                    else:
                        print(f'---- EPOCH {self.epoch} -------------------------')

            elif self.mode=='eval':   # if you where doing eval, after the epoch you go back train
                self.mode='train'
                self.agent.is_training = True
                print(f'---- EPOCH {self.epoch} -------------------------')
                
            else:           #this is in general for mode = 'test'
                self.env.pause()
                will = ""
                while not (will=='y' or will=='n'): 
                    will = str(input('Wanna save? Click "y" for yes or "n" for no:\n -> '))
                if will == 'y':
                    self.plot.save_dict()
                self.env.unpause()
            self.env.unpause()
            self.reset()
            print(f"GOAL:'{self.env.goal_id}' -> {self.env.goal_pos}")
            

        #NEW EPISODE
        #----------------------------------------------------------
        #get commands
        usr_cmd = np.array(twist_to_cmd(data))
        ca_cmd = np.array(self.ca_controller.get_cmd(self.env.pointCloud))
        #ts_cmd = self.ts_controller.get_cmd(self.env.time)
        state_vars = np.hstack([usr_cmd,ca_cmd,self.prev_alpha, self.env.robot.mb_v,self.env.robot.mb_om])
        #print('state vars: ',state_vars)

        #assemble and observe the last episode data
        
        state = [observation,state_vars]
        self.agent.observe(reward,state, done)

        #update time
        t=rospy.get_time()-self.start_time
        dt = round(t - self.time,3)
        self.time = t

        # get alpha from the DDPG    ==> as compute action
        if self.epoch==0 and self.mode == 'train' and self.env.step <= self.hyperParam.warmup:
            #alpha = self.agent.select_action(state)
            if self.random_warmup:
                alpha = self.agent.random_action()
                print(f'Warmup - random: Alpha = {round(alpha,1)}  - dt = {dt} ', end="\r", flush=True)

            else:
                alpha,_ = self.agent.select_action(state)
                print(f'Warmup - true: Alpha = {round(alpha,1)}  - dt = {dt} ', end="\r", flush=True)
        else:
            #print('\n ACT')
            alpha,a_opt = self.agent.select_action(state)
            tag=self.mode
            if self.agent.name == 'ddqn':
                print(f"STEP: {self.env.step} - Alpha = {round(alpha,1)} ({tag}) - A_opt = {round(a_opt,1)} - dt = {dt} ", end="\r", flush=True)

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
        self.plot.store(self.time,usr_cmd,ca_cmd,[0,0],alpha,cmd)
        self.plot.obs_poses[self.env.step]=self.env.pointCloud
        self.plot.ranges[self.env.step]=observation

        if self.verbose:
            self.display_commands_SC(alpha,usr_cmd,ca_cmd,[0,0],cmd)
        
        #print('---------------------')
        #print(' - alpha: ',alpha)
        #print(' - cmd: ',cmd)
        
        self.rate.sleep()


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
            if self.mode == 'train':
                print(' - epsilon: ',self.agent.epsilon)
                print(f' - mem. capacity: {self.agent.buffer.get_capacity()*100}%')
                print(' - mean episode reward: ',self.mean_episode_rewards[-1])
                print(' - mean episode pol. loss: ',self.mean_episode_losses[-1])
            print('---\n')


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
        

    def display_commands_SC(self,alpha,usr_cmd,ca_cmd,cmd):
        print('STEP: ', self.env.step)
        print(' - alpha: ',alpha)
        print(' - user: ',usr_cmd)
        print(' - ca: ',ca_cmd)
        #print(' - ts: ',ts_cmd)
        print('FINAL: ',cmd)
        print('---------------------')

    def display_commands_classic(self,module,cmd):
        print('STEP: ', self.env.step)
        print(' - cmd: ',cmd)
        print(' - module: ', module)
        print('---------------------')

    def shutdownhook(self):
        print('Shout down...')

    def reset(self):
        print('------ RESET -------')
        self.env.reset()
        self.agent.reset(self.env.ls_ranges,np.array([0.0,0.0,0.0,0.0,1.0,0.0,0.0]),1.0)
        self.ts_controller.reset()
        self.pub_goal.publish(String(self.env.goal_id))
        self.episode_reward = 0.
        if not self.plot_dir==None:
            self.plot = Plot(
                goal=self.env.goal_id,
                env = self.env.name,
                parent_dir=self.plot_dir,
                name=self.mode+'epoch_'+str(self.epoch),
                description='')
            self.ca_controller.dir = self.plot.dir

        self.start_time = rospy.get_time()
        



