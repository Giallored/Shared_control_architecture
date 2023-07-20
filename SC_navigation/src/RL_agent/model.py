
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
        self.n_kernels = 16

        #structure functions
        self.conv1 = nn.Conv1d(n_frames,self.n_kernels,kernel_size=5,stride=2)
        self.conv2 = nn.Conv1d(self.n_kernels,1,kernel_size=3,stride=2) 
        self.fc1 = nn.Linear(152, hidden1)
        self.fc2 = nn.Linear(n_states+hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)

        #useful functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.batchNorm1 = nn.BatchNorm1d(self.n_kernels)
        self.batchNorm2 = nn.BatchNorm1d(1)
        self.max_pool = nn.MaxPool1d(3,stride=1)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self,x):
        img,cmd = x
        bs = img.shape[0]
        out = self.relu(self.conv1(img))
        out = self.max_pool(out)
        out = self.batchNorm1(out)
        out = self.relu(self.conv2(out))
        out = self.max_pool(out)
        out = self.batchNorm2(out)
        out = self.relu(self.fc1(out)).reshape(bs,-1)
        out = self.relu(self.fc2(torch.cat([out,cmd],-1)))
        out = self.fc3(out)
        act = torch.sigmoid(out).squeeze(1)
        return act
    
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

        self.n_kernels = 16

        self.conv11 = nn.Conv1d(n_frames,self.n_kernels,kernel_size=5,stride=2)
        self.conv12 = nn.Conv1d(self.n_kernels,1,kernel_size=3,stride=2) 
        self.fc11 = nn.Linear(152, hidden1)
        self.fc12 = nn.Linear(n_states+n_actions+hidden1, hidden2)
        self.fc13 = nn.Linear(hidden2, hidden3)
        self.fc14 = nn.Linear(hidden3,1)

        self.conv21 = nn.Conv1d(n_frames,self.n_kernels,kernel_size=5,stride=2)
        self.conv22 = nn.Conv1d(self.n_kernels,1,kernel_size=3,stride=2) 
        self.fc21 = nn.Linear(152, hidden1)
        self.fc22 = nn.Linear(n_states+n_actions+hidden1, hidden2)
        self.fc23 = nn.Linear(hidden2, hidden3)
        self.fc24 = nn.Linear(hidden3,1)

        self.relu = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm1d(self.n_kernels)
        self.batchNorm2 = nn.BatchNorm1d(1)
        self.max_pool = nn.MaxPool1d(3,stride=1)
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
        bs = img.shape[0]

        out1 = self.relu(self.conv11(img))
        out1 = self.max_pool(out1)
        out1 = self.batchNorm1(out1)
        out1 = self.relu(self.conv12(out1))
        out1 = self.max_pool(out1)
        out1 = self.batchNorm2(out1)
        out1 = self.relu(self.fc11(out1)).reshape(bs,-1)
        out1 = self.relu(self.fc12(torch.cat([out1,cmd,a],-1)))
        out1 = self.fc13(out1)
        out1 = self.layerNorm(out1)
        out1 = self.relu(out1)
        out1 = self.fc14(out1)

        out2 = self.relu(self.conv21(img))
        out2 = self.max_pool(out2)
        out2 = self.batchNorm1(out2)
        out2 = self.relu(self.conv22(out2))
        out2 = self.max_pool(out2)
        out2 = self.batchNorm2(out2)
        out2 = self.relu(self.fc21(out2)).reshape(bs,-1)
        out2 = self.relu(self.fc22(torch.cat([out2,cmd,a],-1)))
        out2 = self.fc23(out2)
        out2 = self.layerNorm(out2)
        out2 = self.relu(out2)
        out2 = self.fc24(out2)

        return out1.squeeze(1),out2.squeeze(1)
    


class Qnet(nn.Module):             # Q-learning network

    def __init__(self, n_states,n_frames, n_actions,hidden1,hidden2,init_w=3e-3):
        super(Qnet, self).__init__()
        self.n_frames = n_frames
        self.n_actions = n_actions
        self.n_kernels = 16

        # architecture
        self.conv1 = nn.Conv1d(n_frames,self.n_kernels,kernel_size=5,stride=2)
        self.conv2 = nn.Conv1d(self.n_kernels,1,kernel_size=3,stride=2) 
        self.fc1 = nn.Linear(41, hidden1)
        self.fc2 = nn.Linear(n_states+hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)        


        self.relu = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm1d(self.n_kernels)
        self.batchNorm2 = nn.BatchNorm1d(1)
        self.max_pool = nn.MaxPool1d(3,stride=1)


    def forward(self,x):
        img,cmd = x
        bs = cmd.shape[0]
        out = self.relu(self.conv1(img))
        out = self.max_pool(out)
        out = self.batchNorm1(out)
        out = self.relu(self.conv2(out))
        out = self.max_pool(out)
        out = self.batchNorm2(out)
        out = self.relu(self.fc1(out)).reshape(bs,-1)
        out = self.relu(self.fc2(torch.cat([out,cmd],-1)))
        out = self.fc3(out)
        return out.squeeze(1)


class Qnet_new(nn.Module):

    def __init__(self, n_states, n_frames, n_actions,hidden1,hidden2):
        super(Qnet_new, self).__init__()
        self.SparseLayer1 = SparseConv(1, 16, 11)
        self.SparseLayer2 = SparseConv(16, 16, 7)
        self.SparseLayer3 = SparseConv(16, 16, 5)
        self.SparseLayer4 = SparseConv(16, 16, 3)
        self.SparseLayer5 = SparseConv(16, 16, 3)
        self.SparseLayer6 = SparseConv(16, 1, 1)


        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(155, hidden1)
        self.fc2 = nn.Linear(n_states+hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, n_actions)

    def forward(self, x):
        
        obs,cmd = x
        n_obs = int(obs.shape[-1]/2)
        print('n_obs: ',n_obs)
        img = obs[:,0,:n_obs].unsqueeze(1)
        mask = obs[:,0,n_obs:].unsqueeze(1)
        bs = cmd.shape[0]
        
        feat, mask = self.SparseLayer1(img, mask)
        feat, mask = self.SparseLayer2(feat, mask)
        feat, mask = self.SparseLayer3(feat, mask)
        feat, mask = self.SparseLayer4(feat, mask)
        feat, mask = self.SparseLayer5(feat, mask)
        feat, mask = self.SparseLayer6(feat, mask)
        print('feat: ',feat.shape)
        
        feat = self.relu(self.fc1(feat)).reshape(bs,-1)
        feat = self.relu(self.fc2(torch.cat([feat,cmd],-1)))
        out = self.fc3(out)
        return x



class SparseConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super().__init__()
        padding = kernel_size//2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels), 
            requires_grad=True)

        self.sparsity = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        kernel = torch.FloatTensor(torch.ones(kernel_size)).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel, 
            requires_grad=False)

        self.relu = nn.ReLU(inplace=True)


        self.max_pool = nn.MaxPool1d(
            kernel_size, 
            stride=1, 
            padding=padding)

        
    def forward(self, x, mask):
        print('x: ',x.shape,'mask: ',mask.shape)
        x = x*mask
        x = self.conv(x)
        #normalizer = 1/(self.sparsity(mask)+1e-8)
        #x = x * normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.relu(x)
        
        mask = self.max_pool(mask)

        return x, mask



