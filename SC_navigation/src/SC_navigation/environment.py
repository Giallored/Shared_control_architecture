import rospy
from std_srvs.srv import Empty
import laser_geometry.laser_geometry as lg
from sensor_msgs.msg import PointCloud2,LaserScan
import numpy as np
from SC_navigation.point_cloud2 import read_points
from std_msgs.msg import Bool,Float64MultiArray
from SC_navigation.utils import from_cmd_msg


class Environment():
    def __init__(self,
                 dt=0.1
                 ):
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_world', Empty) #resets the simulation
        self.pause_sim = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_sim = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.is_running=True
        self.n_step = 0
        self.time=0
        self.obs=[]
        self.dt = dt
        self.laserScanner = LaserScanner()

    def reset(self):
        self.reset_sim()
    
    def pause(self):
        self.pause_sim()

    def unpause(self):
        self.unpause_sim()

    def step(self,action):  
        self.n_step+=1
        self.unpause()
        self.is_running=True
        self.pub_act.publish(action)
        rospy.sleep(self.dt)
        obstacles = self.laserScanner.get_obs_points()
        usr_cmd,ca_cmd,ts_cmd = self.get_commands()
        observation = Observation(self.step)

    def update(self,time,scan_msg):
        self.time=time
        self.obs = self.laserScanner.get_obs_points(scan_msg)






        


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


    def get_obs_points(self,scan_msg=None):
        if scan_msg==None:
            scan_msg=rospy.wait_for_message("scan_raw",LaserScan, timeout=None)
        scan = self.lp.projectLaser(scan_msg)
        points_array=[]
        for p in read_points(scan, skip_nans=True):
            point=[p[0], p[1]]#, data[2], data[3]]
            points_array.append(point)
        return points_array

    






