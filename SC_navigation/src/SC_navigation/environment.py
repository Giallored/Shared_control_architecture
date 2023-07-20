import rospy
from std_srvs.srv import Empty
import numpy as np
import random
from SC_navigation.utils import *
from gazebo_msgs.msg import ModelStates,ModelState
from gazebo_msgs.srv import SetModelState
from SC_navigation.laser_scanner import LaserScanner
from sensor_msgs.msg import LaserScan



class Environment():
    def __init__(self,
                 name:str,
                 robot,
                 n_agents=3,
                 delta_goal=1,
                 delta_coll=0.7,
                 theta_coll=0.7,
                 max_steps=100,
                 verbosity=False
                 ):
        self.name=name
        self.n_agents=n_agents
        self.max_steps=max_steps
        self.verbosity=verbosity
        self.n_actions = n_agents
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_world', Empty) #resets the simulation
        self.pause_sim = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_sim = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.is_running=True
        self.step = 0
        self.laserScanner = LaserScanner()
        self.robot = robot
        self.is_coll=False
        self.is_goal=False
        self.formation_r =3.0
        self.random_warmup=True



        #reward stuff
        self.delta_coll= delta_coll
        self.theta_coll = theta_coll
        self.delta_goal = delta_goal
        self.stacked_steps = 0
        self.stacked_th = 5

        self.prev_act=1.0
        self.cur_act = 1.0
        self.cur_usr_cmd=[0.,0.]
        self.cur_cmd = np.array([0.,0.])
        self.R_safe = rospy.get_param('/rewards/R_safe')  # coeff penalizing sloseness to obstacles
        self.R_col = rospy.get_param('/rewards/R_col')  # const penalizing collisions
        self.R_goal = rospy.get_param('/rewards/R_goal')  #coeff penalizing the distance from the goal
        self.R_end = rospy.get_param('/rewards/R_end')  # const rewarding the rach of the goal
        self.R_alpha = rospy.get_param('/rewards/R_alpha')  #coeff penalizing if the arbitration changes too fast
        self.R_cmd = rospy.get_param('/rewards/R_cmd') 
        
        self.setup()

    def setup(self):
        # read a scan and get all points to compute the current observations
        scan_msg = rospy.wait_for_message('scan_raw', LaserScan, timeout=None)
        self.callback_scan(scan_msg)
        self.n_observations = len(self.observation)

        # read a model state and get the state of the simulation
        model_msg = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=None)
        self.obj_dict,tiago_vels = get_sim_info(model_msg)
        self.robot.set_MBpose(self.obj_dict['tiago'],tiago_vels)
        
        #compute the list of possible goals
        objects =list(self.obj_dict.keys())
        objects.remove('tiago')
        objects.remove('ground_plane')
        self.obs_ids=[]
        self.goal_list=[]
        for id in objects:
            if not id[0:4] == 'wall':
                if id[0:4]=='goal':
                    self.goal_list.append(id)
                else:
                    self.obs_ids.append(id)
        self.goal_id = random.sample(self.goal_list,1)[0]


    def reset(self,mode):
        self.reset_sim()
        self.change_obj_pose(mode)
        #update model poses
        model_msg = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=None)
        self.obj_dict,tiago_vels = get_sim_info(model_msg)
        self.robot.set_MBpose(self.obj_dict['tiago'],tiago_vels)
        print('continue...')
        self.step=0
        self.is_coll=False
        self.is_goal=False

    
    def pause(self):
        self.is_running=False
        self.pause_sim()

    def unpause(self):
        self.is_running=True
        self.unpause_sim()


    def make_step(self):
        self.step += 1
        reward = self.get_reward()
        #is_goal = self.goal_check() 
        is_stuck = self.stuck_check()
        is_end = (self.step>=self.max_steps)
        if self.is_goal or is_stuck or self.is_coll or is_end:
            done=True
        else:
            done = False
        return self.observation, reward, done 
    


    # ---------------------------- CHECKS ---------------------------------
    def goal_check(self,stamp=True):
        goal_dist = np.linalg.norm(np.subtract(self.goal_pos,self.robot.mb_position))
        if goal_dist<=self.delta_goal:
            if stamp:print('\nGoal riched!')
            return True
        else:
            return False
        
    def danger_level(self):
        pos=self.cls_obstacle
        dist = np.linalg.norm(pos)
        theta = np.arctan2(pos[1],pos[0])
        if self.robot.mb_v>0.1:
            if dist<self.delta_coll and abs(theta)<np.arctan2(self.delta_coll,0.2):
                    return 2
            if dist<self.delta_coll*2 and abs(theta)<self.theta_coll:
                return 1
           
        return 0

    def stuck_check(self,stamp=True):
        if self.robot.mb_position == self.robot.prev_mb_position:
            self.stacked_steps += 1
        else:
            self.stacked_steps =0

        if self.stacked_steps >= self.stacked_th:
            if stamp:print('\nIs stucked!')
            self.stacked_steps=0
            return True
        else:
            return False
    
    # ----------------- UPDATES / CALLBACKS -----------------------------
    def update_act(self,act,usr_cmd,cmd):
        self.prev_act=self.cur_act 
        self.cur_act = act
        self.cur_usr_cmd=usr_cmd
        self.cur_cmd = np.array(cmd)

    def callback_robot_state(self,msg):
        t = rospy.get_time()
        if t%0.1==0:
            self.obj_dict,tiago_vel = get_sim_info(msg)
            self.goal_pos = self.obj_dict[self.goal_id].position
            self.robot.set_MBpose(self.obj_dict['tiago'],tiago_vel)   

    def callback_collision(self,data):
        contact=Contact(data)
        obj_1,obj_2=contact.check_contact()
        if not(obj_1==None) and self.is_coll==False and self.is_goal==False:
            if obj_1==self.goal_id:
                self.is_goal=True
                print(f'\nGoal')
            else:
                self.is_coll=True
                print(f'\nCollision: {obj_1} - {obj_2}')

    def callback_scan(self,scan_msg):
        self.laserScanner.scan_msg=scan_msg
        self.observation,self.pointCloud = self.laserScanner.get_obs_points()
        self.cls_obstacle = get_cls_obstacle(self.pointCloud)




    def get_reward(self):

        # safety oriented
        if self.is_coll:
            r_safety = self.R_col
        else:
            r_safety=0
        
        obs_dist = np.linalg.norm(self.cls_obstacle)
        # arbitration oriented
        if self.danger_level()==2:
            r_alpha = self.R_alpha*(1-self.cur_act)      #if safe, must go straigth to the goal
        else:
            r_alpha = self.R_alpha*self.cur_act      #if close to the obstacles, must avoid
        

        # command oriented 
        r_cmd = self.R_cmd*np.linalg.norm(np.subtract(self.cur_cmd,self.cur_usr_cmd))

        # Goal oriented
        goal_dist = np.linalg.norm(np.subtract(self.goal_pos,self.robot.mb_position))
        r_goal = self.R_goal*goal_dist

        if self.is_goal:
            r_end = self.R_end
        else:
            r_end=0

        self.cur_rewards = [r_safety,r_alpha,r_goal,r_cmd,r_end]
        reward = sum(self.cur_rewards)
        
        if False:
            print('------REWARDS-------')
            print(' - r_safe :', r_safety)
            print(' - r_alpha :', r_alpha)
            print(' - r_goal: ',r_goal)
            print(' - r_cmd: ',r_cmd)
            print(' - r_end: ',r_end)
            print('Tot rewards:',reward)
        return reward




    def change_obj_pose(self,mode):
        #for 5 obstacle
        r = 3
        if not r%2==0:
            r_=r+1
        else:
            r_=r
        l = self.obs_ids 
        n = len(l)
        #if self.step%50==0 and self.formation_r>=4.0:
        #    self.formation_r-=0.5

        #change obstacles position
        #ax = np.linspace(-r,r,int(2*r)+1)

        # for 10 obstacles
        #ax_x = np.linspace(2,2*r,int(2*r)+1)
        #ax_y = np.linspace(-r,r,int(2*r))

        #for 5 obstacle
        ax_x = np.linspace(1,r+1,int(r)+1)
        ax_y = np.linspace(-r_/2,r_/2,int(r_)+1)

        #if r%1==0.0: grid.remove([0.0,0.0])
        if mode == 'train':
            grid = [[x,y] for x in ax_x for y in ax_y]
            new_poses = random.sample(grid, n)
        else:
            new_poses = [[1.0,0.5],[1.0,-0.5],[2.25,0.0],[3.5,0.5],[3.5,-0.5]]

        for i in range(n):
            obj_id =l[i]
            state_msg = ModelState()
            new_pos = new_poses[i]
            h=0.3

            state_msg.model_name = obj_id
            state_msg.pose.position.x = new_pos[0]
            state_msg.pose.position.y = new_pos[1]
            state_msg.pose.position.z = h
            #state_msg.pose.orientation.x = 0
            #state_msg.pose.orientation.y = 0
            #state_msg.pose.orientation.z = 0
            #state_msg.pose.orientation.w = 0
            send_state_msg(state_msg)
            

        #change goal position
        r__=r_+1
        ax = np.linspace(-r/2,r/2,int(2*r)+1)
        grid_goal =[
            #*[[x,r_]for x in ax],
            #*[[x,r_] for x in ax],
            *[[r__,y,0.01] for y in ax],
            #*[[0,y] for y in ax]
             ]
        self.goal_pos = random.sample(grid_goal, 1)[0]

        state_msg = ModelState()
        state_msg.model_name = self.goal_id
        state_msg.pose.position.x = self.goal_pos[0]
        state_msg.pose.position.y = self.goal_pos[1]
        state_msg.pose.position.z = h
        send_state_msg(state_msg)


 








