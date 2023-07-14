
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#from ipdb import set_trace as debug

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, n_states, n_frames, n_actions, hidden1=100,hidden2=200, init_w=3e-3,p_drop=0.5):
        super(Actor, self).__init__()
        self.n_actions = n_actions
        self.p_drop = p_drop
        self.n_kernels = 32

        #structure functions
        self.conv1 = nn.Conv1d(n_frames,self.n_kernels,kernel_size=5,stride=2)
        self.conv2 = nn.Conv1d(self.n_kernels,1,kernel_size=3,stride=2) 
        self.fc1 = nn.Linear(126, hidden1)
        self.fc2 = nn.Linear(n_states+hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)

        #useful functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.batchNorm = nn.BatchNorm1d(self.n_kernels)
        self.dropout = nn.Dropout(p=self.p_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self,x):
        img,cmd = x
        out = self.relu(self.conv1(img))
        out = self.batchNorm(out)
        out = self.relu(self.conv2(out))
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(torch.cat([out,cmd],-1)))
        out = self.relu(self.fc3(out))
        print('out pre: ',out )
        out = nn.functional.normalize(input=out, dim=-1)
        print('out post: ',out )

        act = self.softmax(out).squeeze(1)

        return act,out.squeeze(1)
    
    def noisy_softmax(self,z,w):
        if w==None:
            return self.softmax(z)
        else:
            print('z: ',z)
            print('z + w: ',z+w)
            return self.softmax(z+w)



    

class Critic(nn.Module):
    def __init__(self, n_states, n_frames, n_actions, hidden1=128, hidden2=128, hidden3=64, init_w=3e-3):
        super(Critic, self).__init__()

        self.n_kernels = 32

        self.conv11 = nn.Conv1d(n_frames,self.n_kernels,kernel_size=5,stride=2)
        self.conv12 = nn.Conv1d(self.n_kernels,1,kernel_size=3,stride=2) 
        self.fc11 = nn.Linear(126, hidden1)
        self.fc12 = nn.Linear(n_states+n_actions+hidden1, hidden2)
        self.fc13 = nn.Linear(hidden2, hidden3)
        self.fc14 = nn.Linear(hidden3,1)

        self.conv21 = nn.Conv1d(n_frames,self.n_kernels,kernel_size=5,stride=2)
        self.conv22 = nn.Conv1d(self.n_kernels,1,kernel_size=3,stride=2) 
        self.fc21 = nn.Linear(126, hidden1)
        self.fc22 = nn.Linear(n_states+n_actions+hidden1, hidden2)
        self.fc23 = nn.Linear(hidden2, hidden3)
        self.fc24 = nn.Linear(hidden3,1)

        self.relu = nn.ReLU()
        self.batchNorm = nn.BatchNorm1d(self.n_kernels)
        self.layerNorm = nn.LayerNorm(hidden3)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc11.weight.data = fanin_init(self.fc11.weight.data.size())
        self.fc12.weight.data = fanin_init(self.fc12.weight.data.size())
        self.fc21.weight.data = fanin_init(self.fc21.weight.data.size())
        self.fc22.weight.data = fanin_init(self.fc22.weight.data.size())
        self.fc13.weight.data.uniform_(-init_w, init_w)
        self.fc23.weight.data.uniform_(-init_w, init_w)

    
    def forward(self, xs):
        x, a = xs
        img,cmd = x

        out1 = self.relu(self.conv11(img))
        out1 = self.batchNorm(out1)
        out1 = self.relu(self.conv12(out1))
        out1 = self.relu(self.fc11(out1))
        out1 = self.relu(self.fc12(torch.cat([out1,cmd,a],-1)))
        out1 = self.fc13(out1)
        out1 = self.layerNorm(out1)
        out1 = self.relu(out1)
        out1 = self.fc14(out1)

        out2 = self.relu(self.conv21(img))
        out2 = self.batchNorm(out2)
        out2 = self.relu(self.conv22(out2))
        out2 = self.relu(self.fc21(out2))
        out2 = self.relu(self.fc22(torch.cat([out2,cmd,a],-1)))
        out2 = self.fc23(out2)
        out2 = self.layerNorm(out2)
        out2 = self.relu(out2)
        out2 = self.fc24(out2)

        return out1.squeeze(1),out2.squeeze(1)