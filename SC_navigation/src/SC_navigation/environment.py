import rospy
from std_srvs.srv import Empty
import laser_geometry.laser_geometry as lg
from sensor_msgs.msg import PointCloud2,LaserScan
import numpy as np
import random
from SC_navigation.point_cloud2 import read_points
from std_msgs.msg import Bool,Float64MultiArray
from SC_navigation.utils import *
from gazebo_msgs.msg import ModelStates


class Environment():
    def __init__(self,
                 n_agents=3,
                 delta=1,
                 max_steps=1000,
                 verbosity=False
                 ):
        self.n_agents=n_agents
        self.delta = delta
        self.max_steps=max_steps
        self.verbosity=verbosity
        self.n_actions = n_agents
        self.n_states = 100 #to setup
        self.cur_observation = [0]*self.n_states #to setup
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_world', Empty) #resets the simulation
        self.pause_sim = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_sim = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.is_running=True
        self.step = 0
        self.time=0
        self.laserScanner = LaserScanner()
        self.tiago = TIAgo()

        # Once initialized, you need to setup:
        # - n_states (from the points seen by the scan)  
        # - obsjects in the simulations
        # - tiago
        # - goal (chosen random from the objects in the scene)
        self.setup()

        #reward stuff
        self.prev_act=[1.,0.,0.]
        self.cur_act = [1.,0.,0.]
        self.cur_usr_cmd=[0.,0.]
        self.cur_cmd = [0.,0.]
        self.R_safe = -5 # coeff penalizing sloseness to obstacles
        self.R_col = -1000 # const penalizing collisions
        self.R_goal = -0.1 #coeff penalizing the distance from the goal
        self.R_end = 1000 # const rewarding the rach of the goal
        self.R_alpha = 10 #coeff penalizing if the arbitration changes too fast
        self.R_cmd = 1

    def setup(self):
        # read a scan and get all points to compute the current state
        self.cur_observation = self.laserScanner.get_obs_points()
        #get the n_state using the current state
        self.n_states = len(self.cur_observation)
        
        # read the model state to get the tiango and obstacle pose
        self.obj_dict = get_sim_info()
        tiago_pose = self.obj_dict['tiago']
        self.tiago.set_MBpose(tiago_pose)
        obs_ids = list(self.obj_dict.keys())
        obs_ids.remove('tiago')

        
        # get the current goal random
        #self.goal_id = random.sample(obs_ids,1)[0]
        #self.goal_pos = Vec3_to_list(self.obj_dict[self.goal_id].position)


    def reset(self):
        self.reset_sim()
        self.update()
    
    def pause(self):
        self.time=rospy.get_time()
        self.pause_sim()

    def unpause(self):
        self.unpause_sim()

    def update(self):
        #print('UPDATE')
        self.obj_dict = get_sim_info()
        self.goal_pos = Vec3_to_list(self.obj_dict[self.goal_id].position)
        self.tiago.set_MBpose(self.obj_dict['tiago'])        
        self.time = rospy.get_time()
        self.cur_observation = self.laserScanner.get_obs_points()
        self.set_goal_dist()


    def get_response(self):
        observation = self.cur_observation
        reward = self.get_reward()
        if self.goal_dist<=self.delta:# or  self.step>=self.max_steps:
            done=True
        else:
            done = False
        return observation, reward, done 
    
    def update_act(self,act,usr_cmd,cmd):
        self.prev_act=self.cur_act 
        self.cur_act = act
        self.cur_usr_cmd=usr_cmd
        self.cur_cmd = cmd


    def get_reward(self):
        # safety oriented
        cls_obs,min_dist = compute_cls_obs(self.cur_observation)
        if min_dist>=self.delta:
            r_safety = 0
        elif min_dist>self.tiago.clearance and min_dist<self.delta:
            r_safety = (self.delta-min_dist)*self.R_safe
        else:
            r_safety = self.R_col

        # arbitration oriented
        r_alpha = self.R_alpha*np.linalg.norm(np.subtract(self.prev_act,self.cur_act))

        # command oriented 
        r_cmd = self.R_cmd*np.linalg.norm(np.subtract(self.cur_cmd,self.cur_usr_cmd))

        # Goal oriented
        r_goal = self.R_goal*self.goal_dist
        if self.goal_dist<=self.delta:
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
    
    def get_goal_dist(self):
        goal_dist = np.linalg.norm(np.subtract(self.goal_pos,self.tiago.mb_position))
        return goal_dist

    def set_goal_dist(self):
        self.goal_dist = np.linalg.norm(np.subtract(self.goal_pos,self.tiago.mb_position))

    
    def set_goal(self,goal_id:str):
        self.goal_id=goal_id
        try:
            self.goal_pos = Vec3_to_list(self.obj_dict[self.goal_id].position)
        except rospy.ROSInterruptException:
            pass
        







    # here you publish a request for an action to the other nodes
    #  to collect the input commands to arbitrate
    def get_commands(self):
        self.pub_act_request.publish(Bool())
        cmd_msg= rospy.wait_for_message('autonomous_controllers/ts_cmd_vel',Float64MultiArray,timeout=None)
        usr_cmd,ca_cmd,ts_cmd = from_cmd_msg(cmd_msg)




class Observation():
    def __init__(self,step,time,usr_cmd,ca_cmd,ts_cmd,obs):
        self.step=step
        self.time=time
        self.usr_cmd = usr_cmd
        self.ca_cmd = ca_cmd
        self.ts_cmd = ts_cmd
        self.obs=obs
    
    def display(self):
        print('---------------')
        print('STEP: ',self.step)
        print(' - timestep: ',self.time)
        print(' - commands: ',self.step)
        print(' - n_obstacles: ',len(self.obs))
    

class LaserScanner():
    def __init__(self):
        self.lp = lg.LaserProjection()
        self.OFFSET = 20

    def trim(self,scanlist):
        trimmed_scanlist = np.delete(scanlist,range(self.OFFSET),0)
        trimmed_scanlist = np.delete(trimmed_scanlist, range( len(trimmed_scanlist) -self.OFFSET , len(trimmed_scanlist)),0)
        return trimmed_scanlist
    def get_obs_points(self,scan_msg=None):
        if scan_msg==None:
            scan_msg=rospy.wait_for_message("scan_raw",LaserScan, timeout=None)
        scan = self.lp.projectLaser(scan_msg)
        points=[]
        for p in read_points(scan, skip_nans=True):
            point=[p[0], p[1]]#, data[2], data[3]]
            points.append(point)
        trimmed_list=self.trim(points)
        return trimmed_list

    






