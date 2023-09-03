#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist,Point
import numpy as np
from gazebo_msgs.msg import ContactsState,ModelStates
import sys
from SC_navigation.robot_models import TIAgo
import random


#from gazebo_msgs.msg._ContactState.ContactState import Contact
from collections import deque,namedtuple
from std_msgs.msg import Bool,String
from SC_navigation.collision_avoidance import Collision_avoider
from SC_navigation.trajectory_smoother import Trajectory_smooter
import laser_geometry.laser_geometry as lg
from SC_navigation.environment import Environment
from SC_navigation.utils import *
from RL_agent.ddqn import DDQN
from sensor_msgs.msg import LaserScan
import wandb

#roslaunch gazebo_sim tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=static_walls 
#roslaunch SC_navigation start_training.launch mode:=train env:=prova
#roslaunch SC_navigation start_training.launch mode:=train model:=prova-run40 epsilon:=0.8



class Controller():
    
    def __init__(self,mode='test',train_param=None,rate=10,verbose=True):
   
        self.mode = mode
        self.env_name = rospy.get_param('/controller/env')   
        self.rate=rospy.Rate(rate) # 10hz        
        self.verbose=verbose
        
        self.n_agents=3
        self.n_acts = 2

        # folders
        self.output_folder = rospy.get_param('/training/output')
        self.model2load = rospy.get_param('/controller/model')
        self.model_dir = self.get_folder(self.output_folder)
        self.plot_dir = self.get_folder(rospy.get_param('/training/plot_folder'))


        #initializations
        self.epoch=0
        self.hyperParam = train_param
        self.prev_alpha = [1.0,0.0,0.0]
        self.cur_alpha = [1.0,0.0,0.0]
        self.cur_aE = [1.0,0.0,0.0]
        #self.prev_usr_cmd= [0.,0.]
        #self.prev_cmd = [0.,0.]
        self.n_state = self.n_agents*self.n_acts + 3 + 2  #usr_cmd + ca_cmd + prev_alpha + cur_vel
        #self.cur_cmds=[0.0]*self.n_acts*self.n_agents

        #training stuff
        self.episode_reward = 0.0
        self.mean_episode_rewards=[]
        self.mean_episode_losses=[]
        self.max_epochs = self.hyperParam.max_epochs
        self.random_warmup = True
        self.step_warmap= 0
        

        #validation stuff
        self.max_eval_iter = 5
        self.eval_result = (0.0,0.0,0.0)
        

        #instatiations
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
        vals = np.linspace(0.0,1.0,5)
        #self.primitives = [(1,0,0),(0,1,0),(0,0,1)]
        self.primitives = [(x,y,z) for x in vals for y in vals for z in vals if sum((x,y,z))==1.0]

        self.agent = DDQN(self.n_state,self.primitives,self.hyperParam, is_training=(self.mode=='train'))
        self.pub=rospy.Publisher('mobile_base_controller/cmd_vel', Twist, queue_size=1)
        self.pub_goal=rospy.Publisher('goal',String,queue_size=1)
        self.pub_a=rospy.Publisher('alpha',Point,queue_size=1)


        #expert primitives
        self.aE = {1:((1.0,0.0,0.0),'U'),
                2:((1.0,0.0,0.0),'CA'),
                3:((0.5,0.5,0.0),'CA'),
                4:((0.25,0.50,0.25),'CA'),
                5:((0.0,0.75,0.25),'CA')}
    
    def main(self):
        self.start_time = rospy.get_time()
        if not self.model2load =="":
            self.agent.load_weights(self.model_dir)

        if self.mode == 'train':
            wandb.init(project="SharedControlRL",
            config={
            "agent": self.agent.name,
            "environment": "simulator",
            "map":'static_walls',
            "model_dir":self.model_dir,
            "epochs": self.max_epochs,
            "episode_lenght":self.hyperParam.max_episode_length,
            "epsilon":self.agent.epsilon,
            "epsilon_decay":self.agent.epsilon_decay,
            "n_frames":self.hyperParam.n_frames,
            "n_primitives":len(self.primitives),
            "n_state":self.n_state,
            "warmup_episodes":self.hyperParam.warmup,
            "ERB_size":self.hyperParam.rmsize,
            "batch":self.hyperParam.bsize,
        }
        )
        self.epoch = 0

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
        print(f'\n---- EPOCH {self.epoch} -------------------------')
        print(f"GOAL:'{self.env.goal_id}' -> {self.env.goal_pos}\n.\n.\n.\n.\n.\n.\n")
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
        _,_,done = self.env.make_step(self.cur_alpha,self.cur_alpha)
        if done:
            self.reset()
            print('-'*20+'\n.\n.\n.\n.\n.')

        #ca_cmd = self.ca_controller.get_cmd(self.env.cls_obstacle)
        #ts_cmd = self.ts_controller.get_cmd(self.time)
        #cmds = [usr_cmd,ca_cmd,ts_cmd]

        caR_cmd,caT_cmd,_ = self.ca_controller.get_cmd(self.env.cls_obstacle)
        cmds = [usr_cmd,caR_cmd,caT_cmd]

        danger,dist = self.env.danger()
        alpha,tag = self.aE[danger]
        self.cur_alpha = alpha
        #
        self.env.unpause()
        
        
        header = 'STEP ' + str(self.env.step) +' - ' + tag

        if not (self.env.is_goal or self.env.is_coll):
            write_console(header,alpha, alpha ,danger,'-',dt)
        
        cmd = np.sum(np.array(alpha)*np.transpose(cmds),axis=-1)
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)
        #a_msg = Point()
        #a_msg.x = alpha[0]*100
        #a_msg.y = alpha[1]*100
        #a_msg.z = alpha[2]*100
        #self.pub_a.publish(a_msg)

        self.rate.sleep()




    def control_train(self,usr_cmd):
        is_warmup = self.step_warmap <= int(self.hyperParam.warmup)
        dt = self.get_time()

        self.env.pause()
        #assemble and observe the current state
        observation,reward,done = self.env.make_step(self.cur_alpha,self.cur_aE)
        #ca_cmd = self.ca_controller.get_cmd(self.env.cls_obstacle)
        #ts_cmd = self.ts_controller.get_cmd(self.time)
        #cmds = [usr_cmd,ca_cmd,ts_cmd]
        #state_vars = np.hstack([usr_cmd,ca_cmd,self.prev_alpha, self.env.robot.mb_v,self.env.robot.mb_om])

        caR_cmd,caT_cmd = self.ca_controller.get_cmd(self.env.cls_obstacle)
        cmds = [usr_cmd,caR_cmd,caT_cmd]
        state_vars = np.hstack([usr_cmd,caR_cmd,caT_cmd,self.prev_alpha, self.env.robot.mb_v,self.env.robot.mb_om])

        state = [observation,state_vars]
        to_observe = True #self.env.step>10
        self.agent.observe(reward, state, done, save=to_observe)

        if not is_warmup:
            self.agent.update_policy()
        
        self.episode_reward += reward

        if done: self.done_routine()

        danger,dist = self.env.danger()

        aE,_= self.aE[danger]
        self.cur_aE = aE

        #get the new action
        if is_warmup:
            self.step_warmap+=1
            alpha,a_opt = self.agent.select_action(state,aE)
            if self.random_warmup:
                alpha = self.agent.random_action(aE)
                header = 'WARMUP (random) - '+ str(self.step_warmap)
            else:
                header = 'WARMUP (true) - ' + str(self.step_warmap) 
        else:
            alpha,a_opt = self.agent.select_action(state,aE)
            tag='(' + self.mode + ')'
            header = 'STEP ' + str(self.env.step) +' - ' + tag + ' save = '+str(to_observe)

        self.prev_alpha = self.cur_alpha
        self.cur_alpha = alpha

        #if danger==5: alpha =(0.0,1.0,0.0)

        if not (self.env.is_goal or self.env.is_coll):
            write_console(header,alpha,a_opt,danger,self.agent.get_lr(),dt)
        
        # blend commands and send the msg to the robot
        cmd = np.sum(np.array(alpha)*np.transpose(cmds),axis=-1)
        msg = cmd_to_twist(cmd)

        self.env.unpause()
        self.pub.publish(msg)

        #a_msg = Point()
        #a_msg.x = alpha[0]*100
        #a_msg.y = alpha[1]*100
        #a_msg.z = alpha[2]*100
        #self.pub_a.publish(a_msg)

        # store actions & plots
        self.plot.store_act(self.time,usr_cmd,caR_cmd,caT_cmd,alpha,cmd)

        self.rate.sleep()



    def control_test(self,usr_cmd):
        dt = self.get_time()

        #assemble and observe the current state
        observation,reward,done = self.env.make_step(self.cur_alpha)
        self.episode_reward+=reward

        caR_cmd,caT_cmd,cmd_safe = self.ca_controller.get_cmd(self.env.cls_obstacle)
        cmds = [usr_cmd,caR_cmd,caT_cmd]
        state_vars = np.hstack([usr_cmd,caR_cmd,caT_cmd,self.prev_alpha, self.env.robot.mb_v,self.env.robot.mb_om])
        state = [observation,state_vars]

        if done: self.done_routine()
        danger,dist = self.env.danger()

        #get the new action
        _,alpha = self.agent.select_action(state)
        if danger<=2: alpha =(1.0,0.0,0.0)
        self.prev_alpha = self.cur_alpha
        self.cur_alpha = alpha
        tag='(' + self.mode + ')'
        header = 'STEP ' + str(self.env.step) +' - ' + tag

        if not (self.env.is_goal or self.env.is_coll):
            write_console(header,alpha,alpha,danger,'-',dt)
        
        # blend commands and send the msg to the robot
        cmd = np.sum(np.array(alpha)*np.transpose(cmds),axis=-1)

        #if not self.env.safety_check(cmd,dt):
        #    cmd = cmd_safe
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)

        # store actions 
        self.episode_reward+=reward

        a_msg = Point()
        a_msg.x = alpha[0]*100
        a_msg.y = alpha[1]*100
        a_msg.z = alpha[2]*100
        self.pub_a.publish(a_msg)

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
        print('RESET',end='\r',flush=True)
        self.env.reset('random')
        self.agent.reset(self.env.observation,np.zeros(self.n_state),(1.0,0,0))
        #self.ts_controller.reset()
        self.pub_goal.publish(String(self.env.goal_id))
        self.episode_reward = 0.
        if not self.plot_dir==None and not self.mode=='eval':
            self.plot = Plot(
                goal=self.env.goal_id,
                env = self.env.name,
                parent_dir=self.plot_dir,
                type = 'act',
                name=str(self.epoch)+'_epoch')
        self.start_time = rospy.get_time()
    
    def get_time(self):
        t=rospy.get_time()-self.start_time
        dt = round(t - self.time,3)
        self.time = t
        return dt
    


    def done_routine(self):

        self.env.pause()
                  
            
        if self.mode=='train':

            mean_loss = self.agent.episode_loss/self.env.step
            self.mean_episode_losses.append(mean_loss)
            self.report()
            
            epoch = self.epoch

            is_warmup = self.step_warmap <= self.hyperParam.warmup

            #if mean_loss>0.001: #not evaluating
            #    self.train_plot.store_train(self.epoch,self.episode_reward,mean_loss)
            #    self.train_plot.save_dict()
            #    if mean_loss<=min(self.train_plot.mean_loss):
            #        

            if epoch>=self.max_epochs:     #END of TRAINING
                wandb.finish()
                self.pub_goal.publish('END')
                rospy.signal_shutdown('Terminate training')
                tag = 'END'

            elif is_warmup:                     #still  WARMUP
                tag = '(warm) EPOCH '+str(self.epoch)

            else:
                self.agent.save_model(self.model_dir)
                self.plot.save_dict()
                self.epoch+=1

                g_score = self.eval_result[1]*100
                c_score = self.eval_result[2]*100
                r_score = self.eval_result[0]
                wandb.log({"eval_score": r_score,"goals":g_score,"colls":c_score,
                        "loss": self.mean_episode_losses[-1], 
                       "lr":self.agent.get_lr(),"eps":self.agent.epsilon,"beta":self.agent.buffer.beta})

                if epoch>50:
                    self.mode='eval'
                    self.eval_result = EvalResults()

                    tag = 'EVALUATION 0'
                else:
                    tag = '(train) EPOCH '+str(self.epoch)

        elif self.mode=='eval':  
            self.eval_result.register_result(self.episode_reward,goal=self.env.is_goal,coll=self.env.is_coll) 

            #when the evaluation repetition finish
            if self.eval_result.iter > self.max_eval_iter:
                self.eval_result = self.eval_result.get_results()

                #save if the model is fine
                g_score = self.eval_result[1]*100
                c_score = self.eval_result[2]*100
                r_score = self.eval_result[0]
                if g_score >= 80:
                    name = 'e'+str(self.epoch)+'_s'+str(r_score)+'_g'+str(int(g_score))
                    dir = os.path.join(self.model_dir,name)
                    os.makedirs(dir, exist_ok=True)
                    self.agent.save_model(dir)
                
                

                #back to training
                self.mode='train'
                tag = '(train) EPOCH '+str(self.epoch)
            else:
                tag = 'EVALUATION '+str(self.eval_result.iter)
            
        else:           #this is in general for mode = 'test'
            #self.env.pause()
            #will = ""
            #while not (will=='y' or will=='n'): 
            #    will = str(input('Wanna save? Click "y" for yes or "n" for no:\n -> '))
            #if will == 'y':
            #    self.plot.save_dict()
            print('Score = ',self.episode_reward)
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


