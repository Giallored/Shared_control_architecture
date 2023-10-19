
import numpy as np
import rospy
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from collections import deque
from RL_agent.model import DwelingQnet,Qnet
#from RL_agent.ERB import Prioritized_ERB
from RL_agent.ERB import PrioritizedReplayBuffer
from RL_agent.utils import *
from copy import deepcopy
from torch.optim.lr_scheduler import StepLR
from statistics import mean

# from ipdb import set_trace as debug


class DDQN(object):
    def __init__(self, n_states, action_space,args, is_training=True):
        self.name = 'ddqn'
        self.n_states = n_states
        self.n_actions= len(action_space)
        self.action_space = action_space
        print(f'There are {self.n_actions} primitive actions.')
        self.n_frames = args.n_frames
        
        net_cfg = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2
                            }

        self.network = Qnet(self.n_states,self.n_frames, self.n_actions,**net_cfg)
        #self.network = DwelingQnet(self.n_states,self.n_frames, self.n_actions,**net_cfg)
        self.target_network = deepcopy(self.network)


        if is_training:
            #Create replay buffer
            #self.buffer = Prioritized_ERB(self.n_frames,memory_size = args.rmsize)
            self.buffer = PrioritizedReplayBuffer(capacity = args.rmsize,
                                                o_shape = (self.n_frames,313),
                                                s_shape = (self.n_states,), 
                                                a_shape=(1,),
                                                alpha=0.5) 

            # Hyper-parameters
            self.lr = args.rate
            self.batch_size = args.bsize
            self.gamma = args.discount
            self.tau = args.tau
            self.epsilon_decay=args.epsilon_decay
            self.epsilon = args.epsilon
            
            self.epsilon_min=0.01
            self.is_training = is_training
            self.train_iter = 0
            self.max_iter = args.max_train_iter

            self.sync_frequency=200
            self.update_frequency = 5
            #init optimizer

            _optimizer_kwargs = {
                "lr": self.lr,
                "weight_decay": 1e-5,
            }

            self.optimizer  = Adam(self.network.parameters(), **_optimizer_kwargs)
            self.lr_scheduler = StepLR(self.optimizer, step_size=1, gamma=0.9)
            self.scheduler_type = 'StepLR'

            self.epsilon_decay_schedule =lambda n: power_decay_schedule(n, 
                                                        eps_decay = self.epsilon_decay,
                                                        eps_decay_min=self.epsilon_min)

            print(f'Hyper parmaeters are:\n - epsilon = {self.epsilon}\n - epsilon decay = {self.epsilon_decay}\n - learning rate = {self.lr}')


        #initializations
        self.a_t = (1.0,0.0,0.0) # Most recent action
        self.episode_loss=0.
        #self.use_cuda = torch.cuda.is_available()
        self.use_cuda = False
        if self.use_cuda: self.cuda()
    
    def weighted_MSEloss(self,input,target,weights):
        return (weights*torch.pow(input-target,2)).mean()
    

    def reset(self,init_obs,init_cmd,init_act):
        self.observations=deque([init_obs]*self.n_frames,maxlen=self.n_frames)
        self.sVars = init_cmd
        self.a_t=np.array(init_act)
        self.episode_loss=0.

    def get_lr(self):
        return self.lr_scheduler.get_last_lr()[0]
    

    def random_action(self,aE):
        if random.random()>0.3:
            action = random.sample(self.action_space,1)[0] 
        else:
            action = aE
        self.a_t = action
        return action
    
    def select_action(self, s_t,aE=(1.0,0.0,0.0)):
        obs_t,cmd_t=s_t

        #assemble the state
        observation=deepcopy(self.observations)
        observation.append(obs_t)
        obs_tsr = to_tensor(np.array(observation),use_cuda=self.use_cuda).reshape(1,self.n_frames,-1)
        cmd_tsr = to_tensor(cmd_t,use_cuda=self.use_cuda).reshape(1,-1)
        s_tsr = [obs_tsr,cmd_tsr]

        q_tsr = self.network(s_tsr)
        #print('suspect: ',q_tsr)
        #input()
        a_opt = self.action_space[torch.argmax(q_tsr).item()]

        p = np.random.random() #exploration probability


        if self.is_training and p < self.epsilon:   #exploration
            action = self.random_action(aE)
        else:                                       #exploitation
            action = a_opt
        if self.is_training: self.epsilon = self.epsilon_decay_schedule(n=self.train_iter)
        self.a_t = action

        return action,a_opt
    
    def Jtd(self, s, a, r, t, ns, indices, weights ):
        # compute the 'used' Q for  current state using the 'predictor'
        q_val= self.network(s)
        q_val = torch.gather(q_val, 1, a.long()).squeeze(1)

        with torch.no_grad():
            # compute the 'best' action for the next state using the using the 'predictor'
            next_q_val = self.network(ns)
            max_next_q_val = torch.max(next_q_val, 1)[1].unsqueeze(1)

            # compute the 'target' Q using the best act for the next state
            next_q_target_val = self.target_network(ns)
            next_q_val = torch.gather(next_q_target_val,1, max_next_q_val).squeeze(1)

        target_q_val = r + (1 - t) * self.gamma * next_q_val

        # loss computation
        loss = self.weighted_MSEloss(q_val,target_q_val,weights)

        # compute the TD error and update the buffer
        TD_error = abs(target_q_val.detach() - q_val.detach()).cpu()
        TD_error = TD_error.numpy()
        
        self.buffer.update_priorities(indices,TD_error.reshape(-1))
        #self.buffer.update_data(abs(TD_error), indices)
        return loss
    

    def compute_loss(self,batch):
        # Sample batch
        obs_b, svar_b, a_batch, r_b, t_b, next_obs_b, next_svar_b, indices, weights = batch

        obs_tsr = to_tensor(obs_b,use_cuda=self.use_cuda)#.reshape(self.batch_size,self.n_frames,-1)
        svar_tsr = to_tensor(svar_b,use_cuda=self.use_cuda)#.reshape(self.batch_size,-1)
        s_tsr =[obs_tsr,svar_tsr]

        next_obs_tsr = to_tensor(next_obs_b,use_cuda=self.use_cuda)#.reshape(self.batch_size,self.n_frames,-1)
        next_svar_tsr = to_tensor(next_svar_b,use_cuda=self.use_cuda)#.reshape(self.batch_size,-1)
        next_s_tsr = [next_obs_tsr, next_svar_tsr]

        a_tsr = to_long_tensor(a_batch,use_cuda=self.use_cuda)#.reshape(self.batch_size,-1)
        r_tsr = to_tensor(r_b,use_cuda=self.use_cuda)#.squeeze(1)
        t_tsr = to_tensor(t_b.astype(np.float),use_cuda=self.use_cuda)#.squeeze(1)

        weights_tsr = to_tensor(np.array(weights).astype(np.float),use_cuda=self.use_cuda)#.squeeze(1)
        J_td = self.Jtd(s_tsr,a_tsr,r_tsr,t_tsr,next_s_tsr,indices, weights_tsr )

        loss = J_td
        self.episode_loss+=loss.item()
        return loss
    


    def update_policy(self):
        if self.train_iter % self.update_frequency == 0:
            batch = self.buffer.sample(self.batch_size)
            loss = self.compute_loss(batch)
            self.network.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.train_iter%self.sync_frequency:
            soft_update(self.target_network, self.network, self.tau)
        if self.train_iter>self.max_iter:
            rospy.signal_shutdown('Terminate training')
        self.train_iter+=1

    def scheduler_step(self,quantity):
        if self.scheduler_type == 'LambdaLR':
            self.loss_window.append(quantity)
            if len(self.loss_window) == self.window_len:
                self.lr_scheduler.step()
        elif self.scheduler_type == 'ReduceLROnPlateau':
            self.lr_scheduler.step(quantity)
        else:
            self.lr_scheduler.step()



    def eval(self):
        self.network.eval()
        self.target_network.eval()

    def cuda(self):
        print('put in cuda')
        self.network.cuda()
        self.target_network.cuda()


    def observe(self, r_t, s_t1, t_t,save=True):
        obs_t1,sVar_t1=s_t1
        if self.is_training:
            state = [np.array(self.observations),self.sVars]
            self.observations.append(obs_t1)
            self.sVars = sVar_t1
            next_state = [np.array(self.observations),self.sVars]

            if save: 
                #a = tuple(self.a_t)
                #self.buffer.store(state=state, action=self.action_space.index(a),
                #              reward=r_t,done=t_t, next_state=next_state )
                
                self.buffer.store(
                    o = state[0],s = state[1],
                    a = self.action_space.index(tuple(self.a_t)),
                    r = r_t,d = t_t,op = next_state[0],
                    sp = next_state[1])            

            
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

            

