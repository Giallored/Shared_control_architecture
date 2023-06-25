import rospy
from std_srvs.srv import Empty
import numpy as np
import random
from SC_navigation.utils import *
from gazebo_msgs.msg import ModelStates
from SC_navigation.laser_scanner import LaserScanner


class Environment():
    def __init__(self,
                 name:str,
                 robot,
                 n_agents=3,
                 delta_goal=1,
                 delta_coll=0.7,
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
        self.time=0
        self.laserScanner = LaserScanner()
        self.robot = robot
        self.is_coll=False


        #reward stuff
        self.delta_coll = delta_coll
        self.delta_goal = delta_goal
        self.stacked_steps = 0
        self.stacked_th = 5

        self.prev_act=[1.,0.,0.]
        self.cur_act = [1.,0.,0.]
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
        # read a scan and get all points to compute the current state
        self.ls_ranges,self.obstacle_pos = self.laserScanner.get_obs_points()
        self.cur_observation =self.get_observation()

        #get the n_state as: n_ranges + 2 (usr_cmd)
        self.n_states = self.cur_observation.shape[0]
        if self.verbosity: self.env.print_obj_dict()

        # read the model state to get the tiango and obstacle pose
        self.obj_dict = get_sim_info()

        self.robot.set_MBpose(self.obj_dict['tiago'])

        #compute the list of possible goals
        objects =list(self.obj_dict.keys())
        objects.remove('tiago')
        objects.remove('ground_plane')
        self.obs_ids=[]
        self.goal_list=[]
        for id in objects:
            if not id[0:4] == 'wall':
                self.obs_ids.append(id)
                if id[0:4]=='goal':
                    self.goal_list.append(id)
        self.choose_goal()

    def print_obj_dict(self):
        print('Objects in the scene: ')
        for o in self.obj_dict.values():
            print(f"* {o.id}")
            print(f' + position: {o.position}')
            print(f' + orientation: {o.orientation}')
            print(f' + distance: {o.distance}')

    def reset(self):
        self.reset_sim()
        self.step=0
        self.is_coll=False
        self.choose_goal()
        self.update()
    
    def pause(self):
        self.is_running=False
        self.time=rospy.get_time()
        self.pause_sim()

    def unpause(self):
        self.is_running=True
        self.unpause_sim()

    def update(self):
        self.obj_dict = get_sim_info()
        self.goal_pos = self.obj_dict[self.goal_id].position
        self.robot.set_MBpose(self.obj_dict['tiago'])        
        self.time = rospy.get_time()
        self.ls_ranges,self.obstacle_pos = self.laserScanner.get_obs_points()
        self.cur_observation =self.get_observation()
        self.set_goal_dist()

    def make_step(self):
        #observation = self.get_observation()
        reward = self.get_reward()
        is_goal = self.goal_check() 
        is_stuck = self.stuck_check()
        is_end = (self.step>=self.max_steps)
        if is_goal or is_stuck or self.is_coll or is_end:
            done=True
        else:
            done = False
        return self.cur_observation, reward, done 
    
    def goal_check(self,stamp=True):
        if self.goal_dist<=self.delta_goal:
            if stamp:print('\nGoal riched!')
            return True
        else:
            return False
        
    def callback_collision(self,data):
        contact=Contact(data)
        flag,obj_1,obj_2=contact.check_contact()
        if flag and self.is_coll==False:
            self.is_coll=True
            print(f'---\nCollision: {obj_1} - {obj_2}')
  

    #def coll_check(self,stamp=True):
    #    r_pos_2d = self.robot.mb_position[0:2]
    #    min_dist=99999
    #    obj_cls=''
    #    for obj_id in self.obs_ids:
    #        pos_2d = self.obj_dict[obj_id].position[0:2]
    #        dist_i=np.linalg.norm(np.subtract(pos_2d,r_pos_2d))
    #        if dist_i<=self.robot.clear:
    #            if stamp:print(f'Collision with {obj_id}')
    #            return True,None
    #        if dist_i<min_dist:
    #            min_dist=dist_i
    #            obj_cls=obj_id
    #    return False,min_dist
#
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
            
    
    def update_act(self,act,usr_cmd,cmd):
        self.prev_act=self.cur_act 
        self.cur_act = act
        self.cur_usr_cmd=usr_cmd
        self.cur_cmd = np.array(cmd)

    def get_observation(self):
        obs = self.cur_usr_cmd+self.ls_ranges
        obs = np.array(obs,dtype='float64')
        return obs


    def get_reward(self):

        # safety oriented
        if self.is_coll:
            r_safety = self.R_col
        else:
            r_safety=0
            #if min_dist>=self.delta_coll:
            #    r_safety = 0
            #else:
            #    r_safety = (self.delta_coll-min_dist)*self.R_safe
            

        # arbitration oriented
        r_alpha = 0#self.R_alpha*np.linalg.norm(np.subtract(self.prev_act,self.cur_act))

        # command oriented 
        r_cmd = self.R_cmd*np.linalg.norm(np.subtract(self.cur_cmd,self.cur_usr_cmd))

        # Goal oriented
        r_goal = self.R_goal*self.goal_dist

        if self.goal_dist<=self.delta_goal:
            r_end = self.R_end
        else:
            r_end=0

        self.cur_rewards = [r_safety,r_alpha,r_goal,r_cmd,r_end]
        reward = sum(self.cur_rewards)
        if self.verbosity:
            print('r_safety: ',r_safety)
            print('r_alpha: ',r_alpha)
            print('r_goal: ',r_goal)
            print('r_cmd: ',r_cmd)
            print('r_end: ',r_end)
            print('Tot rewards:',reward)
        return reward

    def set_goal_dist(self):
        self.goal_dist = np.linalg.norm(np.subtract(self.goal_pos,self.robot.mb_position))

    def choose_goal(self):
        self.goal_id = random.sample(self.goal_list,1)[0]
        
        #self.goal_id = '20x100cm_cylinder'
        self.goal_pos = self.obj_dict[self.goal_id].position










