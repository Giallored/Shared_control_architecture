import rospy
from geometry_msgs.msg import Twist
import numpy as np
from scipy.spatial.transform import Rotation
import time
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ContactsState

import os
import matplotlib.pyplot as plt
import pickle


class TIAgo():
    def __init__(self, clear=0.2, mb_position=[0.,0.,0.],mb_orientation=[0.,0.,0.]):
        self.mb_position=mb_position # wrt RFworld
        self.prev_mb_position=mb_position
        self.mb_orientation=mb_orientation # wrt RFworld
        self.clear = clear
        self.Tf_tiago_w = np.zeros((4,4))

    def set_MBpose(self,pose):
        self.mb_prev_position=self.mb_position
        self.mb_position=pose.position
        self.mb_orientation=pose.orientation
        rot = Rotation.from_euler('xyz', self.mb_orientation, degrees=False)
        self.Tf_tiago_w = Pose2Homo(rot.as_matrix(),self.mb_position)
        #turn orientation from quat to euler (in rad)
        #rot = Rotation.from_quat(Vec4_to_list(pose.orientation))
        #self.Tf_tiago_w = Pose2Homo(rot.as_matrix(),self.mb_position)
        #self.mb_orientation=rot.as_euler('xyz', degrees=False)
    
    def get_relative_pos(self,obj_pos):
        pos = obj_pos+[1]
        rel_pos = np.dot(np.linalg.inv(self.Tf_tiago_w),pos)
        rel_pos=rel_pos[:-1]
        return rel_pos

def Pose2Homo(rot,trasl):
    p=np.append(trasl,[1])
    M=np.row_stack((rot,[0,0,0]))
    return np.column_stack((M,p))

def quat2euler(quat):
    rot = Rotation.from_quat(quat)
    return rot.as_euler('xyz', degrees=False)


def Vec3_to_list(vector):
    return [vector.x,vector.y,vector.z]
def Vec4_to_list(vector):
    return [vector.x,vector.y,vector.z,vector.w]

def twist_to_list(twist_msg):
    return [twist_msg.linear.x,twist_msg.linear.y,twist_msg.angular.z]

def list_to_twist(linear,angular):
    msg = Twist()
    msg.linear.x = linear[0]
    msg.linear.y = linear[1]
    msg.linear.z = linear[2]
    msg.angular.x = angular[0]
    msg.angular.y = angular[1]
    msg.angular.z = angular[2]
    return msg

def cmd_to_twist(cmd):
    msg = Twist()
    msg.linear.x = cmd[0]
    msg.angular.z = cmd[1]
    return msg

def twist_to_cmd(msg):
    return [msg.linear.x,msg.angular.z]

def blend_commands(w_list,cmd_list,n=3):
    #cmds = np.array_split(cmd_list, n)
    v=0
    om=0
    for i in range(n):
        w_i = w_list[i]
        v_i = cmd_list[i][0]
        v = v_i*w_i
        om_i = cmd_list[i][1]
        om+=w_i*om_i
    return v,om

def compute_cls_obs(obs_list):
        min_distace=math.inf
        cls_point = [25.0,25.0]
        cls_id = 0
        cnt=0
        for obs in obs_list:
            distance = np.linalg.norm(obs)
            if distance<min_distace and distance > 0.0:
                min_distace=distance
                cls_point=obs
                cls_id=cnt
            cnt+=1
        return cls_point,min_distace
        
        
def from_cmd_msg(msg):
    cmds = np.array_split(msg.data, len(msg.data)/2)
    usr_cmd=cmds[0]
    ca_cmd = cmds[1]
    ts_cmd = cmds[2]
    return usr_cmd,ca_cmd,ts_cmd


def countdown(n):
    for i in (range(n,0,-1)): 
        print(f"{i}", end="\r", flush=True)
        time.sleep(1)
    print('GO!')



def get_sim_info():
    ms_msg = rospy.wait_for_message("/gazebo/model_states",ModelStates, timeout=None)
    ids = ms_msg.name
    poses = ms_msg.pose
    tiago_pos = Vec3_to_list( poses[ids.index('tiago')].position)
    dict = {}
    for id,pose in zip(ids,poses):
        pos_i=Vec3_to_list(pose.position)
        theta_i = quat2euler(Vec4_to_list(pose.orientation))
        dist_i = np.linalg.norm(np.subtract(pos_i,tiago_pos))
        obj = Object(id,pos_i,theta_i,dist_i)
        dict[id]=obj

    
    return dict

class Object():
    def __init__(self, id:str, pos,theta, dist:float):
        self.id=id
        self.position=pos
        self.orientation=theta
        self.distance=dist


def shout_down_routine(goal,reward):
    print("Mission accomplised")
    print(' - goal: ', goal)
    print(' - episode reward: ',reward)
    

def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    
    """
    os.makedirs(parent_dir, exist_ok=True)
               
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


class Plot():
    def __init__(self,goal,env,parent_dir,name,description:str=''):
        self.name=name
        self.dir = os.path.join(parent_dir,self.name)
        os.makedirs(self.dir, exist_ok=True)
        self.description=description
        
        self.goal=goal
        self.env=env
        #initializations
        self.timesteps=[0.]
        self.usr_cmd=[[0,0]]
        self.ca_cmd=[[0,0]]
        self.ts_cmd=[[0,0]]
        self.alpha=[[1.0,0.0,0.0]]
        self.cmd=[[0.0,0.0]]
    
    def store(self,t,usr_cmd,ca_cmd,ts_cmd,alpha,cmd):
        self.timesteps=np.append(self.timesteps,t)
        self.usr_cmd=np.append(self.usr_cmd,[usr_cmd],axis=0)
        self.ca_cmd=np.append(self.ca_cmd,[ca_cmd],axis=0)
        self.ts_cmd=np.append(self.ts_cmd,[ts_cmd],axis=0)
        self.alpha=np.append(self.alpha,[alpha],axis=0)
        self.cmd=np.append(self.cmd,[cmd],axis=0)
    
    def save_plot(self,show=False):
        
        f_usr, axs = plt.subplots(2,1, sharey=True)
        axs[0].plot(self.timesteps,self.usr_cmd[:,0])
        axs[0].set_title('linear vel')
        axs[1].plot(self.timesteps,self.usr_cmd[:,1])
        axs[1].set_title('angular vel')
        path = os.path.join(self.dir,'usr_cmd.png')
        plt.savefig(path)

        f_ca, axs = plt.subplots(2,1, sharey=True)
        axs[0].plot(self.timesteps,self.ca_cmd[:,0])
        axs[0].set_title('linear vel')
        axs[1].plot(self.timesteps,self.ca_cmd[:,1])
        axs[1].set_title('angular vel')
        path = os.path.join(self.dir,'ca_cmd.png')
        plt.savefig(path)
        
        f_ts, axs = plt.subplots(2,1, sharey=True)
        axs[0].plot(self.timesteps,self.ts_cmd[:,0])
        axs[0].set_title('linear vel')
        axs[1].plot(self.timesteps,self.ts_cmd[:,1])
        axs[1].set_title('angular vel')
        path = os.path.join(self.dir,'ts_cmd.png')
        plt.savefig(path)

        f_com, axs = plt.subplots(2,1, sharey=True)
        axs[0].plot(self.timesteps,self.cmd[:,1])
        axs[0].set_title('linear vel')
        axs[1].plot(self.timesteps,self.cmd[:,1])
        axs[1].set_title('angular vel')
        path = os.path.join(self.dir,'commands.png')
        plt.savefig(path)

        f_a= plt.plot( self.timesteps,self.alpha)
        plt.legend(['usr_a','ca_a','ts_a'])
        path = os.path.join(self.dir,'alpha.png')
        plt.savefig(path)


        print('Plots saved in: ',self.dir)

        if not self.description=='':
            with open(os.path.join(self.dir,'description.txt'), mode='w') as f:
                f.write(self.description)

        if show:
            plt.show()

    def close(self):
        plt.close('all')

    def save_dict(self):
        dict = {
            'timesteps':self.timesteps,
            'usr_cmd':self.usr_cmd,
            'ca_cmd':self.ca_cmd,
            'ts_cmd':self.ts_cmd,
            'cmd':self.cmd,
            'alpha':self.alpha,
            'env':self.env,
            'goal':self.goal
        }
        where = os.path.join(self.dir,'plot_dict.pkl')
        with open(where, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_dict(self,dict):
        #where = os.path.join(self.dir,dict_name)
        #with open(where, 'rb') as handle:
        #    dict = pickle.load(handle)
        self.timesteps=dict['timesteps']
        self.usr_cmd=dict['usr_cmd']
        self.ca_cmd=dict['ca_cmd']
        self.ts_cmd=dict['ts_cmd']
        self.alpha=dict['alpha']
        self.cmd=dict['cmd']
        
def clamp_angle(theta):
    sign=np.sign(theta)
    theta = (abs(theta)%6.2832)*sign  # 2*np.pi
    if theta>np.pi:theta = theta-2*np.pi
    elif theta<-np.pi:theta = theta+2*np.pi
    return theta

class Contact():
    def __init__(self,msg:ContactsState):
        self.state=msg.states
    
    def check_contact(self):
        if self.state==[]:
            return False,None,None
        else:
            obj_1 = self.clean_name(self.state[0].collision1_name)
            obj_2 = self.clean_name(self.state[0].collision2_name)
            return True,obj_1,obj_2

    def clean_name(self,name):
        final_name=''
        for l in name:
            if l == ':':
                break
            else:
                final_name+=l
        return final_name 
        







