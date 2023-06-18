
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

from RL_agent.model import (Actor, Critic)
from RL_agent.memory import SequentialMemory
from RL_agent.random_process import OrnsteinUhlenbeckProcess
from RL_agent.utils import *

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
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        #use_cuda = torch.cuda.is_available()
        self.use_cuda = False
        if self.use_cuda: self.cuda()

    def init_state(self,state):
        self.s_t=state
        self.a_t=np.array([1.0,0.0,0.0])

    def update_policy(self):
        #print('Update policy...')
        # Sample batch
        #print('Sample:')
        s_batch,a_batch,r_batch,next_s_batch,t_batch = self.memory.sample_and_split(self.batch_size)
        #print(' - size of s_batch: ',s_batch.shape)
        #print(' - size of a_batch: ',a_batch.shape)
        #print(' - size of r_batch: ',r_batch.shape)
        #print(' - size of ns_batch: ',next_s_batch.shape)

        # Prepare for the target q batch
        #print(' - Prepare for the target q batch')
        next_s_tsr = to_tensor(next_s_batch,use_cuda=self.use_cuda, volatile=True)
        t_act_out = self.actor_target(next_s_tsr) #output of the actor target
        next_q_val = self.critic_target([next_s_tsr,t_act_out])
        next_q_val.volatile=False
        
        r_tsr = to_tensor(r_batch,use_cuda=self.use_cuda)
        t_tsr = to_tensor(t_batch.astype(np.float),use_cuda=self.use_cuda)
        target_q_batch = r_tsr + self.discount*t_tsr*next_q_val

        # Critic update
        #print(' - Critic update')
        self.critic.zero_grad()
        a_tsr = to_tensor(a_batch,use_cuda=self.use_cuda)
        s_tsr = to_tensor(s_batch,use_cuda=self.use_cuda)
        q_batch = self.critic([s_tsr,a_tsr])
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        #print(' - Actor update')

        s_tsr = to_tensor(s_batch,use_cuda=self.use_cuda)
        actor_output = self.actor(s_tsr)
        policy_loss = -self.critic([s_tsr,actor_output])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        #print(' - Target update')
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        #print('Valuse loss is ',value_loss)
        #print('Policy loss is ',policy_loss)





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
        #print('Observe:')
        #print(' - state size:',self.s_t.shape)
        #print(' - act size:',self.a_t.shape)
        #print(' - reward size:',r_t.shape)
        if self.is_training:
            self.memory.append(self.s_t, self.a_t, r_t, done) #append sample in memory
            self.s_t = s_t1 #update curr state

    def random_action(self):
        # using the Dirichlet distrution to get the action to sum to 1
        #action = np.random.uniform(-1.,1.,self.nb_actions)
        action = np.random.dirichlet(np.ones(3),size=1)[0]
        self.a_t = action
        return action

    def select_action(self, s_t, decay_epsilon=True):
        #get the action from the actor
        s_tsr = to_tensor(np.array([s_t]),use_cuda=self.use_cuda)
        a_tsr = self.actor(s_tsr)
        action = to_numpy(a_tsr,self.use_cuda).squeeze(0)
        if self.is_training:
            #insert some noise
            noise= np.random.dirichlet(np.ones(3),size=1)[0]
            e= max(self.epsilon, 0)
            #action += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
            action = (1.0-e)*action+e*noise
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    def reset(self, obs):
        self.s_t = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )

    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
