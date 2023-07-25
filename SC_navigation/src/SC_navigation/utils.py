import rospy
from geometry_msgs.msg import Twist
import numpy as np
from scipy.spatial.transform import Rotation
import time
from gazebo_msgs.msg import ContactsState
from gazebo_msgs.srv import SetModelState

import sys
import os
import matplotlib.pyplot as plt
import pickle



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

def get_cls_obstacle(poin_cloud):
        dist_list = [np.linalg.norm(p) for p in poin_cloud]
        indices = np.argsort(dist_list)
        sorted_cloud =poin_cloud[indices]
        try :
            return sorted_cloud[0]
        except:
            return np.array([9999,9999])
        


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

        
def send_state_msg(msg):
    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state(msg)
    except:
        print("Service call failed: %s")

def get_sim_info(ms_msg):
    ids = ms_msg.name
    #poses
    poses = ms_msg.pose
    vels = ms_msg.twist
    tiago_pos = Vec3_to_list( poses[ids.index('tiago')].position)
    dict = {}
    for id,pose in zip(ids,poses):
        pos_i=Vec3_to_list(pose.position)
        theta_i = quat2euler(Vec4_to_list(pose.orientation))
        dist_i = np.linalg.norm(np.subtract(pos_i,tiago_pos))
        obj = Object(id,pos_i,theta_i,dist_i)
        dict[id]=obj

    #tiago vels
    v_abs = Vec3_to_list(vels[ids.index('tiago')].linear)
    om = vels[ids.index('tiago')].angular.z
    v = np.linalg.norm(v_abs)
    return dict,[v,om]



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
    def __init__(self,goal,env,parent_dir,name='',type='act'):
        self.name=name
        self.type = type
        if type == 'act':
            self.dir = os.path.join(parent_dir,self.name)
            os.makedirs(self.dir, exist_ok=True)
            #initializations
            self.timesteps=[]
            self.usr_cmd=[]
            self.ca_cmd=[]
            self.ts_cmd=[]
            self.alpha=[]
            self.cmd=[]
            self.obs_poses={}
            self.ranges = {}
        else:
            self.dir = parent_dir   
            self.epochs = []
            self.rewards = []
            self.mean_loss = []
        self.goal=goal
        self.env=env
        
    
    def store_act(self,t,usr_cmd,ca_cmd,ts_cmd,alpha,cmd):
        self.timesteps.append(t)
        self.usr_cmd.append(usr_cmd)
        self.ca_cmd.append(ca_cmd)
        self.ts_cmd.append(ts_cmd)
        self.alpha.append(alpha)
        self.cmd.append(cmd)

    def store_train(self,epoch,reward,loss):
        self.epochs.append(epoch)
        self.rewards.append(reward)
        self.mean_loss.append(loss)
       

    def close(self):
        plt.close('all')

    def save_dict(self):
        if self.type=='act':
            dict = {
                'type':self.type,
                'timesteps':self.timesteps,
                'usr_cmd':self.usr_cmd,
                'ca_cmd':self.ca_cmd,
                'ts_cmd':self.ts_cmd,
                'cmd':self.cmd,
                'alpha':self.alpha,
                'env':self.env,
                'goal':self.goal,
                'obs':self.obs_poses,
                'ranges':self.ranges
            }
            where = os.path.join(self.dir,'plot_dict.pkl')

        else:
            dict = {
                'type':self.type,
                'epoch':self.epochs,
                'reward':self.rewards,
                'loss':self.mean_loss
            }
            where = os.path.join(self.dir,'train_dict.pkl')

        with open(where, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Plots saved in: ',where)
    

    def load_dict(self,dict):
        #where = os.path.join(self.dir,dict_name)
        #with open(where, 'rb') as handle:
        #    dict = pickle.load(handle)
        self.type = dict['type']
        if self.type=='act':
            self.timesteps=dict['timesteps']
            self.usr_cmd=dict['usr_cmd']
            self.ca_cmd=dict['ca_cmd']
            self.ts_cmd=dict['ts_cmd']
            self.alpha=dict['alpha']
            self.cmd=dict['cmd']
        else:
            self.epoch = dict['epoch']
            self.reward = dict['reward']
            self.loss = dict['loss']
        
def clamp_angle(theta):
    while theta>np.pi:
        theta-=2*np.pi
    while theta<-np.pi:
        theta+=2*np.pi
    return theta


def write_console(header,alpha,a_opt,danger,lr,dt):
    l = [header,
        ' - Alpha = ' + str(alpha),
        ' - Alpha_opt = ' + str(a_opt),
        ' - Danger lev = ' + str(danger),
        ' - laerning rate = ' + str(lr),
        ' - dt = ' + str(dt)]
    
    for _ in range(len(l)):
        sys.stdout.write("\x1b[1A\x1b[2K") # move up cursor and delete whole line
    for i in range(len(l)):
        sys.stdout.write(l[i] + "\n") # reprint the lines


class Contact():
    def __init__(self,msg:ContactsState):
        self.state=msg.states
    
    def check_contact(self):
        if self.state==[]:
            return None,None
        else:
            obj_1 = self.clean_name(self.state[0].collision1_name)
            obj_2 = self.clean_name(self.state[0].collision2_name)
            return obj_1,obj_2

    def clean_name(self,name):
        final_name=''
        for l in name:
            if l == ':':
                break
            else:
                final_name+=l
        return final_name 


#for processing pointclouds to images
def pc2img(pc,defi=2,height=300,width=400):
    pc = np.around(pc,decimals=defi)*10**defi#clean
    pc[:,1]+=width/2      #translate
    #print((pc[:,0]>=0) & (pc[:,0]<=300)&(pc[:,1]>=0) & (pc[:,1]<=height))
    pc = pc[(pc[:,0]>0)  & (pc[:,0]<height)  & (pc[:,1]>0)  & (pc[:,1]<width)] #crop
    pc=np.array(pc).astype('int') 
    rows = pc[:,0]
    cols = pc[:,1]
    img = np.zeros((height,width))
    img[rows,cols]=255
    kernel = np.ones((5,5),np.uint8)#
    img = cv.dilate(img,kernel,iterations = 1)
    #img = cv.resize(img, (400,300), interpolation = cv.INTER_AREA)
    return img






