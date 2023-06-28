
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

# from ipdb import set_trace as debug

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, nb_states, nb_actions, args):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions= nb_actions
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_target = Actor(self.nb_states, self.nb_actions, **net_cfg)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.prate)

        self.critic = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **net_cfg)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.rate)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = SequentialMemory(limit=args.rmsize, window_length=args.window_length)
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.epsilon_decay=args.epsilon_decay
        self.epsilon = args.epsilon
        self.is_training = args.is_training
        self.n_frame=args.n_frame

        # Hyper-parameters for the dynamcally changing SD of the noise (Gaussian)
        self.sigma_w = 0.3 # noise SD 
        self.sigma_th = 0.3 # threshold on the noise SD
        self.dist_avg = 0.2 #avg distance between 
        self.dist_d = 0.3  #desired distance between a_opt and a_noisy
        self.dist_d_decay=0.999
        self.sm_dist = 0.8 # smoothing factors in the exponential moving avg for the distance
        self.sm_sd = 0.5 # smoothing factors in the exponential moving avg for the SD
        self.sf = 1.01 #scaling factor

        self.sigma_w_target = 0.1 # noise SD for the target 


        #initializations
        self.a_t = [1.0,0.0,0.0] # Most recent action
        self.episode_value_loss=0.
        self.episode_policy_loss=0.

        #use_cuda = torch.cuda.is_available()
        self.use_cuda = False
        if self.use_cuda: self.cuda()
    
    def update_hp(self):
        self.epsilon*=self.epsilon_decay
        self.dist_d*=self.dist_d_decay
        


    def reset(self,state):
        self.state=deque([state]*self.n_frame,maxlen=self.n_frame)
        self.s_t=state
        self.a_t=np.array([1.0,0.0,0.0])
        self.episode_value_loss=0.
        self.episode_policy_loss=0.

    def update_policy(self):
        print('UPDATE')
        # Sample batch
        s_batch,a_batch,r_batch,next_s_batch,t_batch = self.memory.sample_and_split(self.batch_size)
        # Prepare for the target q batch

        # compute target actions
        with torch.no_grad(): 
            next_s_tsr = to_tensor(next_s_batch,use_cuda=self.use_cuda)#, volatile=True)
            target_a_raw = self.actor_target(next_s_tsr) #output of the actor target
            w = np.random.normal(0, self.sigma_w_target, self.batch_size) #noise
            target_a_tsr = self.actor_target.noisy_softmax(target_a_raw,w)

        #compute target
        self.critic_target.zero_grad()
        next_q_val = self.critic_target([next_s_tsr,target_a_tsr])
        r_tsr = to_tensor(r_batch,use_cuda=self.use_cuda)
        t_tsr = to_tensor(t_batch.astype(np.float),use_cuda=self.use_cuda)
        target_q_batch = r_tsr + self.discount*t_tsr*next_q_val

        # Critic update
        self.critic.zero_grad()
        a_tsr = to_tensor(a_batch,use_cuda=self.use_cuda)
        #s_tsr = to_tensor(s_batch,use_cuda=self.use_cuda)
        s_tsr = to_tensor(np.array(s_batch),use_cuda=self.use_cuda).reshape(self.batch_size,-1)
        q_batch = self.critic([s_tsr,a_tsr])
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        #s_tsr = to_tensor(s_batch,use_cuda=self.use_cuda)
        s_tsr = to_tensor(np.array(s_batch),use_cuda=self.use_cuda).reshape(self.batch_size,-1)
        a_raw = self.actor(s_tsr)
        a_opt = self.actor.softmax(a_raw)
        policy_loss = -self.critic([s_tsr,a_opt])
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        #store the losses
        self.episode_value_loss+=value_loss.item()
        self.episode_policy_loss+=policy_loss.item()







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
        if self.is_training:
            self.memory.append(np.array(self.state), self.a_t, r_t, done) #append sample in memory
            self.state.append(s_t1)
            #self.memory.append(self.s_t, self.a_t, r_t, done) #append sample in memory
            #self.s_t = s_t1 #update curr state

    def random_action(self):
        # using the Dirichlet distrution to get the action to sum to 1
        #action = np.random.uniform(-1.,1.,self.nb_actions)
        action = np.random.dirichlet(np.ones(3),size=1)[0]
        self.a_t = action
        return action

    def select_action(self, s_t):

        #assemble the state
        state=deepcopy(self.state)
        state.append(s_t)
        s_tsr = to_tensor(np.array(state),use_cuda=self.use_cuda).reshape(-1)

        #get the optimal action
        a_raw = self.actor(s_tsr)
        a_opt = self.actor.softmax(a_raw)

        if self.is_training:
            w = np.random.normal(0, self.sigma_w, 1)  #Gaussian Noise
            a_noise = self.actor.noisy_softmax(a_raw,w)
            action = to_numpy(a_noise,self.use_cuda)
            self.update_noise(a_opt,a_noise)

        else:
            action = to_numpy(a_opt,self.use_cuda)

        return action



        #get the action from the actor
        #s_tsr = to_tensor(np.array([s_t]),use_cuda=self.use_cuda)
        
        #a_tsr = 
        #action = to_numpy(a_tsr,self.use_cuda)#.squeeze(0)
        #print(f'------\n Alpha_before: {action}\n')
        #if self.is_training:
        #    #insert some noise
        #    noise= np.random.dirichlet(np.ones(3),size=1)[0]
        #    e= max(self.epsilon, 0)
        #    #action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        #    action =  self.is_training*((1.0-e)*action+e*noise)
        #action = np.clip(action, -1., 1.)       
        #self.a_t = action
        
    
    
    def update_noise(self,a_opt,a_noise):
        # compute the avg distance between a_opt and a_noise
        self.dist_avg = (1-self.sm_dist)*self.avg_dist + self.sm_dist*np.linalg.norm(a_opt-a_noise)

        #update the scaled variance
        if self.sigma_w < self.sigma_th:
            #variance decreases/increases as the distace is higher/lower than the desired distace.
            self.variance = self.sigma_w * (self.sf**np.sign(self.dist_d-self.dist_avg))
        else:
            self.variance = self.sigma_th

        #update the SD
        self.sigma_w = (1-self.sm_sd )*self.sigma_w + self.sm_sd*self.variance

        

            




    def load_weights(self, output_dir):
        if output_dir is None: return
        print('LOAD MODEL: ',output_dir)

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
