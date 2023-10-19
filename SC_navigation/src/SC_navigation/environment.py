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
                 rate=10,
                 n_agents=3,
                 n_obstacles=3,
                 delta_goal=1,
                 delta_coll=0.7,
                 theta_coll=0.7,
                 max_steps=100,
                 verbosity=False
                 ):
        self.last_time = 0
        self.name=name
        self.rate=rate
        self.n_agents=n_agents
        self.max_steps=max_steps
        self.verbosity=verbosity
        self.n_actions = n_agents
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_world', Empty) #resets the simulation
        self.pause_sim = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_sim = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.is_running=True
        self.step = 0
        self.cnt=0
        self.time = rospy.get_time()
        self.laserScanner = LaserScanner()
        self.robot = robot
        self.is_coll=False
        self.is_goal=False
        self.random_warmup=True

        #map stuff
        self.map_size = [5,3,0.3]
        self.obs_clear = 1.5
        self.n_obstacles = n_obstacles
        self.obstacles_pos = [] 

        #reward stuff
        self.delta_coll= delta_coll
        self.theta_coll = theta_coll
        self.delta_goal = delta_goal
        self.stacked_steps = 0
        self.stacked_th = 5

        self.last_gDist=0

        self.R_safe = rospy.get_param('/rewards/R_safe')  # coeff penalizing sloseness to obstacles
        self.R_col = rospy.get_param('/rewards/R_col')  # const penalizing collisions
        self.R_goal = rospy.get_param('/rewards/R_goal')  #coeff penalizing the distance from the goal
        self.R_end = rospy.get_param('/rewards/R_end')  # const rewarding the rach of the goal
        self.R_alpha = rospy.get_param('/rewards/R_alpha')  #coeff penalizing if the arbitration changes too fast
        
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
        self.goal_pos = self.obj_dict[self.goal_id].position
        self.n_obstacles = len(self.obs_ids)
        for o_id in self.obs_ids:
            o_pos = self.obj_dict[o_id].position
            self.obstacles_pos.append(o_pos)
        


    def reset(self,type='random',shuffle=True):
        self.reset_sim()
        if shuffle:self.change_obj_pose(type)
        #update model poses
        model_msg = rospy.wait_for_message('gazebo/model_states', ModelStates, timeout=None)
        self.obj_dict,tiago_vels = get_sim_info(model_msg)
        self.robot.set_MBpose(self.obj_dict['tiago'],tiago_vels)
        self.last_gDist =  np.linalg.norm(np.subtract(self.goal_pos[0:2],self.robot.mb_position[0:2]))
        #print('continue...')
        self.step=0
        self.is_coll=False
        self.is_goal=False

    
    def pause(self):
        self.is_running=False
        self.pause_sim()

    def unpause(self):
        self.is_running=True
        self.unpause_sim()


    def make_step(self,alpha,alphaE=[0,0,0]):
        self.step += 1
        reward = self.get_reward(alpha,alphaE)
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
        
    def danger(self):
        pos=self.cls_obstacle
        dist = np.linalg.norm(pos)
        theta = np.arctan2(pos[1],pos[0])
        theta_th = np.arcsin(0.35/np.clip(dist,0.35,np.inf))
        if dist >=1.5:
            danger = 1
        elif dist >=1.0:
            danger = 2
        elif dist >=0.7:
            danger = 3
        elif dist <0.7:
            if abs(theta)>theta_th:
                danger = 4
            else:
                danger = 5 
        
        return danger,dist,theta


    def safety_check(self,cmd,dt):
        X_obs_0 = self.cls_obstacle
        v = cmd[0]
        om = cmd[1]
        theta = om*dt
        d_X = [v*np.cos(theta)*dt, v*np.sin(theta)*dt]
        X_obs_1 = np.subtract(X_obs_0,d_X)
        theta_1 = clamp_angle(np.arctan2(X_obs_1[1],X_obs_1[0]))   
        dist_1 = np.linalg.norm(X_obs_1)

        if theta_1<=np.arcsin(0.35/np.clip(dist_1,0.35,np.inf)) and dist_1<0.3:
            #print('NOT SAFE')
            return False
        else:
            return True
        


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

    def callback_robot_state(self,msg):
        
        self.cnt+=1
        if self.cnt%50==0:
            self.obj_dict,tiago_vel = get_sim_info(msg)
            self.goal_pos = self.obj_dict[self.goal_id].position
            self.robot.set_MBpose(self.obj_dict['tiago'],tiago_vel)   
            

    def callback_collision(self,data):
        contact=Contact(data)
        obj_1,obj_2=contact.check_contact()
        if not(obj_1==None) and self.is_coll==False and self.is_goal==False:
            #check for GOAL
            if obj_1==self.goal_id or obj_2==self.goal_id:
                self.is_goal=True
                print(f'*Goal*\n')
            #check for COLLISIONS
            else:
                self.is_coll=True
                print(f'*Collision: {obj_1} - {obj_2}*\n')

    def callback_scan(self,scan_msg):
        self.laserScanner.scan_msg=scan_msg
        self.observation,self.pointCloud = self.laserScanner.get_obs_points()
        self.cls_obstacle = get_cls_obstacle(self.pointCloud)



    def get_reward(self,alpha,alphaE=[0,0,0]):

        # safety oriented
        #e = np.linalg.norm(self.cls_obstacle)
#
        #if self.is_coll:
        #    r_safety = self.R_col
        #elif e<1.0:
        #    r_safety=self.R_safe/e
        #else:
        #    r_safety=0

        x,y=self.cls_obstacle
        dist = np.sqrt(x**2+y**2)

        if self.is_coll:
            r_safety = self.R_col
        elif dist<1.0:
            bear = np.arctan2(y,x)
            close = (1.0 - dist) / (1.0 - 0.35) +0.001
            angl = (np.pi - abs(bear))/np.pi
            r_safety=self.R_safe*angl*close
        else:
            r_safety=0
        
        # arbitration oriented
        r_alpha = self.R_alpha * alpha[0]#+ (alpha==alphaE)*1
        
        # Goal oriented
        g_dir = np.subtract(self.goal_pos[0:2],self.robot.mb_position[0:2])
        g_bear = np.arctan2(g_dir[1],g_dir[0])-self.robot.mb_position[2]
        g_dist = np.linalg.norm(g_dir)
        delta_gDist = self.last_gDist - g_dist
        self.last_gDist = g_dist
        
        if self.is_goal:
            r_goal = self.R_end
        else:
            r_goal=self.R_goal*delta_gDist*(np.pi - abs(g_bear))

        self.cur_rewards = [r_safety,r_alpha,r_goal]
        reward = sum(self.cur_rewards)
        
        if False:
            print('------REWARDS-------')
            print(' - r_goal :', r_goal)
            print(' - r_safe :', r_safety)
            print(' - r_alpha :', r_alpha)
            print('Tot rewards:',reward)
        return reward




    def change_obj_pose(self,type ='random'):
        r_x,r_y,h= self.map_size 
        x_min = 1.0
        x_max = r_x+1.0
        y_min = -r_y/2
        y_max = r_y/2
        
        if type == 'random':
            self.obstacles_pos=[]
            for i in range(self.n_obstacles):
                clear = False
                while not clear:
                    obs_i = np.array([random.uniform(x_min, x_max),random.uniform(y_min,y_max)])
                    clear = all(np.linalg.norm(obs_i-np.array(o)) > self.obs_clear for o in self.obstacles_pos)
                self.obstacles_pos.append(obs_i.tolist()) 

        #else:
        #    ax_x = np.linspace(x_min,x_max,int(2*(x_max-x_min))+1)
        #    ax_y = np.linspace(y_min,y_max,int(2*(y_max-y_min))+1)
        #    grid = [[x,y] for x in ax_x for y in ax_y]
        #    self.obstacles_pos = random.sample(grid, self.n_obstacles+1)
        
        self.goal_pos = [x_max+1,random.uniform(y_min,y_max)]
        self.obstacles_pos.append(self.goal_pos)

        ids = self.obs_ids + [self.goal_id]
        

        #new_poses = [[1.4671088789204418, -0.09558857286171785], [3.4393445430910483, -1.4883161455188647], [2.400304106133938, 0.6591271214720171], [3.6625928010018756, 0.17565946755608364], [1.0475654234293055, -1.2300963785508037], [6.0, -0.11139969122068472]]
        #ids = ['20x100cm_cylinder', '20x100cm_cylinder_0', '20x100cm_cylinder_1', '20x100cm_cylinder_2', '20x100cm_cylinder_3', 'goal']

        for i in range(len(ids)):
            obj_id =ids[i]
            state_msg = ModelState()
            new_pos = self.obstacles_pos[i]

            state_msg.model_name = obj_id
            state_msg.pose.position.x = new_pos[0]
            state_msg.pose.position.y = new_pos[1]
            state_msg.pose.position.z = h
            send_state_msg(state_msg)
            



 








