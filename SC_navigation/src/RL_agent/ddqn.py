
import numpy as np
import rospy
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from collections import deque
from RL_agent.model import Qnet,Qnet_new
from RL_agent.ERB import Prioritized_ERB
from RL_agent.utils import *
from copy import deepcopy
from torch.nn import Softmax

# from ipdb import set_trace as debug


class DDQN(object):
    def __init__(self, n_states, n_frames, n_actions, args):
        self.name = 'ddqn'
        self.n_states = n_states
        self.n_actions= n_actions
        self.action_space = list(np.linspace(0,1,n_actions))
        self.n_frames = n_frames
        self.lr = 0.0001
        
        # Create Actor and Critic Network
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2
                            }

        self.network = Qnet(self.n_states,self.n_frames, self.n_actions,**net_cfg)
        self.target_network = deepcopy(Qnet(self.n_states,self.n_frames, self.n_actions,**net_cfg))

        self.optim  = Adam(self.network.parameters(), lr=self.lr)
        self.loss= nn.MSELoss()

        hard_update(self.target_network, self.network) # Make sure target is with the same weight
        
        #Create replay buffer
        self.buffer = Prioritized_ERB(self.n_frames,memory_size = args.rmsize)
        
        # Hyper-parameterssigma
        self.batch_size = args.bsize
        self.gamma = args.discount
        self.tau = args.tau
        self.discount = args.discount
        self.epsilon_decay=0.999
        self.epsilon = 0.7
        self.epsilon_min=0.1
        self.is_training = args.is_training
        self.policy_freq=2 #delayed actor update


        self.train_iter = 0
        self.sync_frequency=200
        self.max_iter = args.max_train_iter

        #initializations
        self.a_t = 1.0 # Most recent action
        self.episode_loss=0.

        #use_cuda = torch.cuda.is_available()
        self.use_cuda = False
        if self.use_cuda: self.cuda()
    
    def update_hp(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay

    def reset(self,init_obs,init_cmd,init_act):
        self.observations=deque([init_obs]*self.n_frames,maxlen=self.n_frames)
        self.sVars = init_cmd
        self.a_t=np.array(init_act)
        self.episode_loss=0.
        self.update_hp()

    def e_greedy(self):
        p = np.random.random()
        if p < self.epsilon:
            return 'explore'
        else:
            return 'exploit'
    

    def random_action(self):
        action = np.random.choice(self.action_space)
        self.a_t = action
        return action
    
    def select_action(self, s_t):
        obs_t,cmd_t=s_t

        #assemble the state
        observation=deepcopy(self.observations)
        observation.append(obs_t)
        obs_tsr = to_tensor(np.array(observation),use_cuda=self.use_cuda).reshape(1,self.n_frames,-1)
        cmd_tsr = to_tensor(cmd_t,use_cuda=self.use_cuda).reshape(1,-1)
        s_tsr = [obs_tsr,cmd_tsr]

        q_tsr = self.network(s_tsr)
        a_opt = self.action_space[torch.argmax(q_tsr).item()]

        p = np.random.random() #exploration probability


        if self.is_training and p < self.epsilon:   #exploration
            action = self.random_action()
        else:                                       #exploitation
            action = a_opt
        
        self.a_t = action

        return action,a_opt
    



    def update_policy(self):
        self.train_iter+=1
        # Sample batch

        obs_b, svar_b, a_batch, r_b, t_b, next_obs_b, next_svar_b, indices, weights = self.buffer.sample(self.batch_size)

        obs_tsr = to_tensor(obs_b,use_cuda=self.use_cuda)#.reshape(self.batch_size,self.n_frames,-1)
        svar_tsr = to_tensor(svar_b,use_cuda=self.use_cuda)#.reshape(self.batch_size,-1)
        s_tsr =[obs_tsr,svar_tsr]

        next_obs_tsr = to_tensor(next_obs_b,use_cuda=self.use_cuda)#.reshape(self.batch_size,self.n_frames,-1)
        next_svar_tsr = to_tensor(next_svar_b,use_cuda=self.use_cuda)#.reshape(self.batch_size,-1)
        next_s_tsr = [next_obs_tsr, next_svar_tsr]

        a_tsr = torch.LongTensor(a_batch)#.reshape(self.batch_size,-1)
        r_tsr = to_tensor(r_b,use_cuda=self.use_cuda).squeeze(1)
        t_tsr = to_tensor(t_b.astype(np.float),use_cuda=self.use_cuda).squeeze(1)

        # compute target Q
        with torch.no_grad():
            next_q = self.target_network.forward(next_s_tsr)
            next_q_max = torch.max(next_q, dim=-1).values
            target_q = r_tsr + (1 - t_tsr)*self.gamma*next_q_max
        
        # compute Q
        q = self.network(s_tsr)
        q = torch.gather(q, 1, a_tsr).squeeze(1)

        # loss computation
        loss = self.loss(q,target_q)
        self.episode_loss+=loss.item()

        # backpropagation
        with torch.no_grad():
            loss_copy = loss.detach()
            weight = sum(np.multiply(weights, loss_copy))
        loss *= weight

        self.network.zero_grad()
        loss.backward()
        self.optim.step()


        # compute the TD error and update the buffer
        
        TD_error = abs(target_q.detach() - q.detach()).numpy()
        self.buffer.update_data(abs(TD_error), indices)
        

        if self.train_iter%self.sync_frequency:
            soft_update(self.target_network, self.network, self.tau)

        if self.train_iter>self.max_iter:
            rospy.signal_shutdown('Terminate training')

            #print('policy loss: ',policy_loss.item())
        




        #store the losses
        


    def eval(self):
        self.network.eval()
        self.target_network.eval()

    def cuda(self):
        print('put in cuda')
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()

    def observe(self, r_t, s_t1, t_t):
        obs_t1,sVar_t1=s_t1
        if self.is_training:
            state = [np.array(self.observations),self.sVars]
            self.observations.append(obs_t1)
            self.sVars = sVar_t1
            next_state = [np.array(self.observations),self.sVars]

            self.buffer.store(state=state, action=self.action_space.index(self.a_t),
                              reward=r_t,done=t_t, next_state=next_state )
            

            
    def load_weights(self, output_dir):

        if output_dir is None: return
        
        ('LOAD MODEL: ',output_dir)

        self.network.load_state_dict(
            torch.load('{}/q_network.pkl'.format(output_dir))
        )
        self.target_network.load_state_dict(
            torch.load('{}/q_network.pkl'.format(output_dir))
        )

    def save_model(self,output_dir):
        print('SAVE MODEL')
        torch.save(
            self.network.state_dict(),
            '{}/q_network.pkl'.format(output_dir)
        )

