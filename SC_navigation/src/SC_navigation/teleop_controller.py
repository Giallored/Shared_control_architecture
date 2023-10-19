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
    
    def __init__(self,mode='test',hyper_param=None,rate=10,verbose=True):
   
        self.mode = mode
        self.env_name = rospy.get_param('/controller/env')   
        self.rate=rospy.Rate(rate) # 10hz        
        self.verbose=verbose
        
        self.n_agents=3
        self.n_acts = 2

        # folders
        self.model2load = rospy.get_param('/controller/model')
        self.output_folder = rospy.get_param('/training/output')
        self.parent_dir = rospy.get_param('/training/parent_dir')
        self.weights_dir = os.path.join(self.parent_dir,self.output_folder)
        self.model_dir = os.path.join(self.weights_dir,self.model2load)
        self.hyperParam = hyper_param


        #initializations
        self.prev_alpha = [1.0,0.0,0.0]
        self.cur_alpha = [1.0,0.0,0.0]
        self.cur_aE = [1.0,0.0,0.0]
        self.n_state = self.n_agents*self.n_acts + 3 + 2  #usr_cmd + ca_cmd + prev_alpha + cur_vel        

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
                               max_steps=10000,
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
        if not self.model2load =="":
            self.agent.load_weights(self.model_dir)

        print('Controller node is ready!')
        print(' - Mode: ',self.mode)
        print(f' - Model: {self.model2load}')

        #-------Setup
        self.reset()
        rospy.Subscriber('robot_bumper',ContactsState,self.env.callback_collision)
        rospy.Subscriber('gazebo/model_states',ModelStates,self.env.callback_robot_state)
        rospy.Subscriber('scan_raw',LaserScan,self.env.callback_scan)
        rospy.Subscriber('usr_cmd_vel',Twist,self.callback)

        rospy.on_shutdown(self.shutdownhook)
        rospy.spin()

    

    def callback(self, usr_msg):
        usr_cmd =np.array(twist_to_cmd(usr_msg)) #direct control

        if self.mode=='direct':
            self.pub.publish(usr_msg)

        elif self.mode=='classic':
            self.control_classic(usr_cmd)

        elif self.mode =='test':
            self.agent.is_training = False
            self.control_test(usr_cmd)

        else:
            rospy.ROSInterruptException
            print('Mode is NOT supported!')


    def control_classic(self,usr_cmd):
        _,reward,done = self.env.make_step(self.cur_alpha,self.cur_alpha)
        self.episode_reward+=reward
        if done: self.done_routine()

        caR_cmd,caT_cmd,_ = self.ca_controller.get_cmd(self.env.cls_obstacle)
        cmds = [usr_cmd,caR_cmd,caT_cmd]

        danger,dist = self.env.danger()
        alpha,tag = self.aE[danger]
        self.cur_alpha = alpha
        #
        self.env.unpause()
        
        header = 'STEP ' + str(self.env.step) +' - ' + tag

        
        cmd = np.sum(np.array(alpha)*np.transpose(cmds),axis=-1)
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)
        #a_msg = Point()
        #a_msg.x = alpha[0]*100
        #a_msg.y = alpha[1]*100
        #a_msg.z = alpha[2]*100
        #self.pub_a.publish(a_msg)

        self.rate.sleep()


    def control_test(self,usr_cmd):
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


        # blend commands and send the msg to the robot
        cmd = np.sum(np.array(alpha)*np.transpose(cmds),axis=-1)

        if not self.env.safety_check(cmd,0.1):
            cmd = cmd_safe
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)

        a_msg = Point()
        a_msg.x = alpha[0]*100
        a_msg.y = alpha[1]*100
        a_msg.z = alpha[2]*100
        self.pub_a.publish(a_msg)

        self.rate.sleep()

    def shutdownhook(self):
        print('Shout down...')

    def reset(self):
        print('RESET',end='\r',flush=True)
        self.env.reset(shuffle=True,type='random')
        self.agent.reset(self.env.observation,np.zeros(self.n_state),(1.0,0,0))
        self.episode_reward=0

    def done_routine(self):
        
        print('----')
        print('Score = ',self.episode_reward)
        print('Steps = ',self.env.step)
        
        self.reset()
    


