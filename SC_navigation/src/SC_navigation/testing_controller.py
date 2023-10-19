#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist,Point
import numpy as np
from gazebo_msgs.msg import ContactsState,ModelStates
from SC_navigation.robot_models import TIAgo


#from gazebo_msgs.msg._ContactState.ContactState import Contact
from collections import deque,namedtuple
from std_msgs.msg import String
from SC_navigation.collision_avoidance import Collision_avoider
from SC_navigation.trajectory_smoother import Trajectory_smooter
import laser_geometry.laser_geometry as lg
from SC_navigation.environment import Environment
from SC_navigation.utils import *
from RL_agent.ddqn import DDQN
from RL_agent.utils_testing import *
from sensor_msgs.msg import LaserScan
import wandb

#roslaunch gazebo_sim tiago_gazebo.launch public_sim:=true end_effector:=pal-gripper world:=static_walls 
#roslaunch SC_navigation start_testing.launch mode:=test model:=static_random_walls-run2 repeats:=20 
#e120_s526_g83.0


Result = namedtuple('Result',field_names=['score','n_steps','alpha_data','ending'])

class Controller():
    
    def __init__(self,mode='classic',model_dir = '',test_dir = '',train_param=None,repeats = 10,rate=10,shuffle=True,verbose=True):

        self.mode = mode
        self.rate=rospy.Rate(rate) # 10hz        
        self.verbose=verbose
        self.repeats = int(repeats)
        self.n_agents=rospy.get_param('/controller/n_agents')
        self.n_acts = rospy.get_param('/controller/n_acts')

        # folders
        self.test_dir = test_dir
        self.results = {}
        dir = os.path.join(self.test_dir,'results.txt')
        self.result_txt =open(dir,"w+")
        self.model_dir = model_dir

        self.cnt_model = 0
        self.iter_model = 0
        self.shuffle = shuffle

        if self.mode =='test':
            self.model_list = sorted(os.listdir(self.model_dir),reverse=True)
        
        
        

        
        #initializations
        self.hyperParam = train_param
        self.prev_alpha = [1.0,0.0,0.0]
        self.cur_alpha = [1.0,0.0,0.0]
        self.cur_aE = [1.0,0.0,0.0]
        self.n_state = self.n_agents*self.n_acts + 3 + 2  #usr_cmd + ca_cmd + prev_alpha + cur_vel

        #training stuff
        #self.episode_reward = 0.0
        #self.mean_episode_rewards=[]
        #self.mean_episode_losses=[]
        #self.max_epochs = self.hyperParam.max_epochs
        #self.random_warmup = True
        #self.step_warmap= 0
        

        #instatiations
        self.ca_controller = Collision_avoider(
            delta=rospy.get_param('/controller/delta_coll'),
            K_lin=rospy.get_param('/controller/K_lin'),
            K_ang=rospy.get_param('/controller/K_ang'),
            k_r=rospy.get_param('/controller/k_ac'),
            )
        env_name = 'test_random'
        self.ts_controller = Trajectory_smooter()
        self.tiago = TIAgo(clear =rospy.get_param('/controller/taigoMBclear'))
        self.env = Environment(env_name,n_agents=self.n_agents,
                               rate=rate,
                               delta_coll= rospy.get_param('/controller/delta_coll'),
                               theta_coll = rospy.get_param('/controller/theta_coll'),
                               delta_goal= rospy.get_param('/training/delta_goal'),
                               max_steps=self.hyperParam.max_episode_length,
                               robot = self.tiago)
        
        vals = np.linspace(0.0,1.0,5)
        #self.primitives = [(1,0,0),(0,1,0),(0,0,1)]
        self.primitives = [(x,y,z) for x in vals for y in vals for z in vals if sum((x,y,z))==1.0]
        self.agent = DDQN(self.n_state,self.primitives,self.hyperParam, is_training=False)
        self.pub=rospy.Publisher('mobile_base_controller/cmd_vel', Twist, queue_size=1)
        self.pub_goal=rospy.Publisher('goal',String,queue_size=1)
        self.pub_a=rospy.Publisher('alpha',Point,queue_size=1)

        #expert primitives
        self.aE = {1:((1.0,0.0,0.0),'U'),
                2:((1.0,0.0,0.0),'CA'),
                3:((0.5,0.5,0.0),'CA'),
                4:((0.25,0.50,0.25),'CA'),
                5:((0.0,0.75,0.25),'CA')}

#---------------------CHECK UNTIL NOW
    def main(self):
        self.start_time = rospy.get_time()
        
        if self.mode == "test":
            self.model_name = self.load_next_model()

        print('Controller node is ready!')
        print(' - Mode: ',self.mode)
        print(' - Directory: ',self.model_dir)
        print(' - Repeats: ',self.repeats)

        #-------Setup
        self.reset()
        rospy.Subscriber('robot_bumper',ContactsState,self.env.callback_collision)
        rospy.Subscriber('gazebo/model_states',ModelStates,self.env.callback_robot_state)
        rospy.Subscriber('scan_raw',LaserScan,self.env.callback_scan)
        rospy.Subscriber('usr_cmd_vel',Twist,self.callback)

        rospy.on_shutdown(self.shutdownhook)
        #print(f'\n---- MODEL {self.cnt_model} - iter {self.iter_model} -------------------------\n{self.model_name}\n.\n.\n.\n.\n.\n')
        if self.mode =='test':print(f'\n---- MODEL {self.cnt_model} {self.model_name}')
        else: print('\n'+'-'*20)
        print(f'Iter: {self.iter_model}')
        
        self.time=self.start_time

        rospy.spin()

    def load_next_model(self):
        model_name = self.model_list.pop()
        model2load = os.path.join(self.model_dir,model_name)
        self.agent.load_weights(model2load)
        self.cnt_model +=1
        self.iter_model = 0
        self.results = {}
        return model_name

    def callback(self, usr_msg):
        usr_cmd =np.array(twist_to_cmd(usr_msg)) #direct control
        if self.mode=='classic' or self.mode=='const':
            self.control_classic(usr_cmd)
        elif self.mode =='test':
            self.agent.is_training = False
            self.control_test(usr_cmd)
        else:
            rospy.ROSInterruptException
            print('Mode is NOT supported!')


    def control_classic(self,usr_cmd):
        dt = self.get_time()
        _, reward,done = self.env.make_step(self.cur_alpha,self.cur_alpha)
        if done: self.done_routine()
        self.episode_reward+=reward

        caR_cmd,caT_cmd,_ = self.ca_controller.get_cmd(self.env.cls_obstacle)
        cmds = [usr_cmd,caR_cmd,caT_cmd]

        danger,dist,bear = self.env.danger()
        if  self.mode=='classic':
            alpha,tag = self.aE[danger]
        else:
            alpha = (1/3,1/3,1/3)
            tag = 'CONST'
        self.cur_alpha = alpha
        
        header = 'STEP ' + str(self.env.step) +' - ' + tag

        #if not (self.env.is_goal or self.env.is_coll):
        #    write_console(header,alpha, alpha ,danger,'-',dt)
        
        self.episode_data.append(alpha)
        cmd = np.sum(np.array(alpha)*np.transpose(cmds),axis=-1)
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)
        a_msg = Point()
        a_msg.x = alpha[0]*100
        a_msg.y = alpha[1]*100
        a_msg.z = alpha[2]*100
        self.pub_a.publish(a_msg)

        self.rate.sleep()


    def control_test(self,usr_cmd):
        dt = self.get_time()

        #assemble and observe the current state
        observation,reward,done = self.env.make_step(self.cur_alpha)
        self.episode_reward+=reward
        caR_cmd,caT_cmd,_ = self.ca_controller.get_cmd(self.env.cls_obstacle)
        cmds = [usr_cmd,caR_cmd,caT_cmd]
        state_vars = np.hstack([usr_cmd,caR_cmd,caT_cmd,self.prev_alpha, self.env.robot.mb_v,self.env.robot.mb_om])
        state = [observation,state_vars]
        

        if done: self.done_routine()
        danger,dist,bear = self.env.danger()

        #get the new action
        _,alpha = self.agent.select_action(state)
        if danger<=2: alpha =(1.0,0.0,0.0)
        self.prev_alpha = self.cur_alpha
        self.cur_alpha = alpha
        tag='(' + self.mode + ')'
        header = 'STEP ' + str(self.env.step) +' - ' + tag

        #if not (self.env.is_goal or self.env.is_coll):
        #    write_console(header,alpha,alpha,danger,'-',dt)
        
        # blend commands and send the msg to the robot
        cmd = np.sum(np.array(alpha)*np.transpose(cmds),axis=-1)
        msg = cmd_to_twist(cmd)
        self.pub.publish(msg)

        # store actions 
        self.episode_data.append(alpha)

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
        #print('RESET',end='\r',flush=True)
        self.env.reset('random',shuffle=self.shuffle)
        self.agent.reset(self.env.observation,np.zeros(self.n_state),(1.0,0,0))
        self.pub_goal.publish(String(self.env.goal_id))
        self.episode_reward = 0.0

        self.episode_data = []
        
        self.start_time = rospy.get_time()
    
    def get_time(self):
        t=rospy.get_time()-self.start_time
        dt = round(t - self.time,3)
        self.time = t
        return dt
    


    def done_routine(self):

        self.env.pause()
        ending = 'goal'*self.env.is_goal + 'coll'*self.env.is_coll + 'None'*(1-self.env.is_goal)*(1-self.env.is_coll)
        if self.mode=='test':  
            result_i = Result(score=self.episode_reward,
                              n_steps=self.env.step,
                              alpha_data=self.episode_data,
                              ending=ending)
            print(f' - score = {self.episode_reward}\n - steps = {self.env.step}')
            self.results[self.iter_model] = result_i
            self.iter_model+=1

            if self.iter_model == self.repeats:
                #save data from the current tested model
                dir = os.path.join(self.test_dir,self.model_name)
                os.makedirs(dir, exist_ok=True)
                file_name = os.path.join(dir,"result.txt")
                f= open(file_name,"w+")
                f.write('PARAMS:\n * seed = None (uses clock)\n * %d iterations\n * max n. steps = 200\n * 0.1 random action by the user\r\n'%self.repeats)
                success_rate,score = show_results(self.results,
                                                 modules = ['usr','ca_r','ca_t'],
                                                 repeats = self.repeats,
                                                 file_name = f,
                                                 mode='eval',verbose=True)
                f.close()
                self.result_txt.write(self.model_name + ' : ' + ' (' + str(score) + ') ' + str(success_rate) + '\n')
                
                if not self.model_list:
                    self.result_txt.close()
                    self.pub_goal.publish('END')
                    rospy.signal_shutdown('Terminate training')
                    tag = 'END'
                self.model_name = self.load_next_model()
                print('\n'+'-'*20)
                print(f'MODEL {self.cnt_model} {self.model_name}')

            tag = self.model_name
        else:
            self.iter_model+=1
            result_i = Result(score=self.episode_reward,
                              n_steps=self.env.step,
                              alpha_data=self.episode_data,
                              ending=ending)
            self.results[self.iter_model] = result_i
            tag = 'classic'
            
            if self.iter_model == self.repeats:
                #save data from the current tested model
                file_name = os.path.join(self.test_dir,"result.txt")
                f= open(file_name,"w+")
                f.write('PARAMS:\n * seed = None (uses clock)\n * %d iterations\n * max n. steps = 200\n * 0.1 random action by the user\r\n'%self.repeats)
                success_rate,score = show_results(self.results,
                                                 modules = ['usr','ca_r','ca_t'],
                                                 repeats = self.repeats,
                                                 file_name = f,
                                                 mode='eval',verbose=True)
                f.close()

                self.pub_goal.publish('END')
                rospy.signal_shutdown('Terminate training')
        
        self.env.unpause()
        self.reset()
        print('--')
        print(f'Iter: {self.iter_model}')
        #print(f'\n------------- Model {self.cnt_model} - iter {self.iter_model} -----------------')
        #print(tag+'\n.\n.\n.\n.\n')
    



 


