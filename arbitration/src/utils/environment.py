import rospy
from std_srvs.srv import Empty
import laser_geometry.laser_geometry as lg
from sensor_msgs.msg import PointCloud2,LaserScan
import numpy as np
from auto_controller.point_cloud2 import read_points
from arbitration.laser_scanner import LaserScanner
from std_msgs.msg import Bool,Float64MultiArray



class Environment():
    def __init__(self,
                 pub_act_request,
                 dt=0.1
                 ):
        self.reset_sim = rospy.ServiceProxy('/gazebo/reset_world', Empty) #resets the simulation
        self.pause_sim = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.unpause_sim = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.is_running=True
        self.n_step = 0
        self.pub_act_request = pub_act_request #publisher topic
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
        observation = Observation(self.step,)



        


    # here you publish a request for an action to the other nodes
    #  to collect the input commands to arbitrate
    def get_commands(self):
        self.pub_act_request.publish(Bool())
        cmd_msg= rospy.wait_for_message('autonomous_controllers/ts_cmd_vel',Float64MultiArray,timeout=None)
        usr_cmd,ca_cmd,ts_cmd = from_cmd_msg(cmd_msg)

        
        
def from_cmd_msg(msg):
    cmds = np.array_split(msg.data, len(msg.data)/2)
    usr_cmd=cmds[0]
    ca_cmd = cmds[1]
    ts_cmd = cmds[2]
    return usr_cmd,ca_cmd,ts_cmd


class Observation():
    def __init__(self,step,time,usr_cmd,ca_cmd,ts_cmd,obs):
        self.step=step
        self.time=time
        self.usr_cmd = usr_cmd
        self.ca_cmd = ca_cmd
        self.ts_cmd = ts_cmd
        self.cmds=cmds
        self.obs=obs
    
    def display(self):
        print('---------------')
        print('STEP: ',self.step)
        print(' - timestep: ',self.time)
        print(' - commands: ',self.step)
        print(' - n_obstacles: ',len(self.obs))
    


    






