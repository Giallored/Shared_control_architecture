
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from collections import deque
from RL_agent.model import (Actor, Critic)
from RL_agent.memory import SequentialMemory
from RL_agent.random_process import OrnsteinUhlenbeckProcess
from RL_agent.utils import *
from copy import deepcopy
from torch.nn import Softmax

# from ipdb import set_trace as debug


class DDPG(object):
    def __init__(self, n_states, n_frames, n_actions, args):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.n_states = n_states
        self.n_actions= n_actions
        self.n_frames = n_frames
        
        # Create Actor and Critic Network
        net_cfg_actor = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w,
            'p_drop': 0.5
        }

        net_cfg_critic = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.n_states,self.n_frames, self.n_actions, **net_cfg_actor)
        self.actor_target = Actor(self.n_states, self.n_frames, self.n_actions, **net_cfg_actor)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)
        self.loss= nn.MSELoss()
        self.softmax = Softmax(dim=-1)

        self.critic = Critic(self.n_states, self.n_frames, self.n_actions, **net_cfg_critic)
        self.critic_target = Critic(self.n_states, self.n_frames, self.n_actions, **net_cfg_critic)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=n_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.sigma_decay=args.sigma_decay
        self.sigma_w = args.sigma # noise SD 
        self.noise_clip = 1.0
        self.is_training = args.is_training
        self.policy_freq=2 #delayed actor update
        self.max_train_iter=args.train_iter
        self.train_iter = 0

        # Hyper-parameters for the dynamcally changing SD of the noise (Gaussian)
        self.variance = self.sigma_w
        self.sigma_th = 1.0 # threshold on the noise SD
        self.dist_avg = 0.2 #avg distance between 
        self.dist_d = 0.1  #desired distance between a_opt and a_noisy
        self.dist_d_decay=0.9999
        self.sm_dist = 0.8 # smoothing factors in the exponential moving avg for the distance
        self.sm_sd = 0.8 # smoothing factors in the exponential moving avg for the SD
        self.sf = 1.1 #scaling factor

        self.sigma_w_target = 0.1 # noise SD for the target 


        #initializations
        self.a_t = [1.0,0.0,0.0] # Most recent action
        self.episode_value_loss=0.
        self.episode_policy_loss=0.

        #use_cuda = torch.cuda.is_available()
        self.use_cuda = False
        if self.use_cuda: self.cuda()
    
    def update_hp(self):
        #self.epsilon*=self.epsilon_decay
        self.dist_d*=self.dist_d_decay
        self.actor.p_drop*=0.99
        


    def reset(self,init_state):
        init_obs,init_cmd = init_state
        self.observations=deque([init_obs]*self.n_frames,maxlen=self.n_frames)
        self.cmds = init_cmd
        self.a_t=np.array([1.0,0.0,0.0])
        self.episode_value_loss=0.
        self.episode_policy_loss=0.

    def update_policy(self):
        self.train_iter+=1
        # Sample batch
        obs_batch,svar_batch,a_batch,r_batch,next_obs_batch,next_svar_batch,t_batch = self.memory.sample_and_split(self.batch_size)
       
        obs_tsr = to_tensor(obs_batch,use_cuda=self.use_cuda).reshape(self.batch_size,self.n_frames,-1)
        svar_tsr = to_tensor(svar_batch,use_cuda=self.use_cuda).reshape(self.batch_size,1,-1)
        s_tsr =[obs_tsr,svar_tsr]

        next_obs_tsr = to_tensor(next_obs_batch,use_cuda=self.use_cuda).reshape(self.batch_size,self.n_frames,-1)
        next_svar_tsr = to_tensor(next_svar_batch,use_cuda=self.use_cuda).reshape(self.batch_size,1,-1)
        next_s_tsr = [next_obs_tsr, next_svar_tsr]

        a_tsr = to_tensor(a_batch,use_cuda=self.use_cuda).reshape(self.batch_size,1,-1)
        r_tsr = to_tensor(r_batch,use_cuda=self.use_cuda)
        t_tsr = to_tensor(t_batch.astype(np.float),use_cuda=self.use_cuda)

        with torch.no_grad():
            # compute target actions
            _,z_target = self.actor_target(next_s_tsr) # target action
            epsilon = torch.normal(mean=0, std=0.7,size= z_target.shape).clamp(-self.noise_clip, self.noise_clip)#1))   #noise
            target_a_tsr = self.softmax(z_target+epsilon).unsqueeze(1)
            #print('target_a: ',target_a_tsr.squeeze())
            #print('original: ',self.actor_target(next_s_tsr, None).squeeze())
        
            #compute target
            target_q1,target_q2 = self.critic_target([next_s_tsr,target_a_tsr])  # target Q_val
            target_q_val = torch.min(target_q1, target_q2)
            target_y = r_tsr + self.discount*t_tsr*target_q_val

        # Value update
        y1,y2 = self.critic([s_tsr,a_tsr])  #current value stimate
        print('--')
        #print(f'y1 = {to_numpy(y1,self.use_cuda).squeeze(0)},y2 = {to_numpy(y2,self.use_cuda).squeeze(0)}')
        value_loss = self.loss(y1,target_y) + self.loss(y2,target_y) #loss
        value_loss.backward()
        
        self.critic.zero_grad()
        self.critic_optim.step()
        self.episode_value_loss+=value_loss.item()
        #print('value loss: ',value_loss.item())



        # Delayed policy updates
        if self.train_iter % self.policy_freq == 0:

            a_opt,_ = self.actor(s_tsr)
            a_opt=a_opt.reshape(self.batch_size,1,-1)
            q_opt,_=self.critic([s_tsr,a_opt])
            #print(f'q_opt = {to_numpy(q_opt,self.use_cuda).squeeze(0)}')
            policy_loss = -q_opt.mean()   #loss
            self.episode_policy_loss+=policy_loss.item()

            self.actor.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

            # Target update
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)
            #print('policy loss: ',policy_loss.item())



        #store the losses
        


    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()


    def cuda(self):
        print('put in cuda')
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, done):
        obs_t1,cmd_t1=s_t1
        if self.is_training:
            self.memory.append(state=[np.array(self.observations),self.cmds],
                               action = self.a_t, reward = r_t, terminal =done) #append sample in memory
            self.observations.append(obs_t1)
            self.cmds = cmd_t1
            #self.memory.append(self.s_t, self.a_t, r_t, done) #append sample in memory
            #self.s_t = s_t1 #update curr state

    def random_action(self):
        # using the Dirichlet distrution to get the action to sum to 1
        #action = np.random.uniform(-1.,1.,self.nb_actions)
        action = np.random.dirichlet(np.ones(3),size=1)[0]
        self.a_t = action
        return action

    def select_action(self, s_t):
        obs_t,cmd_t=s_t
        #assemble the state
        observation=deepcopy(self.observations)
        observation.append(obs_t)
        
        obs_tsr = to_tensor(np.array(observation),use_cuda=self.use_cuda).reshape(1,self.n_frames,-1)
        cmd_tsr = to_tensor(cmd_t,use_cuda=self.use_cuda).reshape(1,1,-1)
        s_tsr = [obs_tsr,cmd_tsr]
        #get the optimal action (no noise)
        a_opt,z_opt = self.actor(s_tsr)
        #print('\n---')
        #print('a_opt: ',a_opt)
        #print('sigma: ',self.sigma_w)
        
        if self.is_training:
            epsilon = torch.normal(mean=0, std=self.sigma_w,size=z_opt.shape)#1))  #Gaussian Noise
            a_noise = self.softmax(z_opt+epsilon)
            self.update_noise(a_opt.detach(),a_noise.detach())
            action = to_numpy(a_noise,self.use_cuda).squeeze(0)
            
        else:
            action = to_numpy(a_opt,self.use_cuda).squeeze(0)
        self.a_t = action

        return action,to_numpy(a_opt,self.use_cuda).squeeze(0)

    
    
    def update_noise(self,a_opt,a_noise):

        ## compute the avg distance between a_opt and a_noise
        ##self.dist_avg = (1-self.sm_dist)*self.dist_avg + self.sm_dist*np.linalg.norm(a_opt-a_noise)
        #self.act_dist = np.linalg.norm(a_opt-a_noise)
        ##update the scaled variance
        #if self.sigma_w < self.sigma_th:
        #    #variance decreases/increases as the distace is higher/lower than the desired distace.
        #    self.variance = self.sigma_w * (self.sf**np.sign(self.dist_d-self.act_dist))
        #else:
        #    self.variance = self.sigma_th
#
        ##update the SD
        #self.sigma_w = (1-self.sm_sd )*self.sigma_w + self.sm_sd*self.variance

        self.sigma_w *=self.sigma_decay

        self.update_hp()

            




    def load_weights(self, output_dir):

        if output_dir is None: return
        
        ('LOAD MODEL: ',output_dir)

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output_dir))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output_dir))
        )


    def save_model(self,output_dir):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output_dir)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output_dir)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if self.use_cuda:
            torch.cuda.manual_seed(s)
