
import os
import torch
from torch.autograd import Variable

#USE_CUDA = False#torch.cuda.is_available()
#FLOAT = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

def prRed(prt): print("\033[91m {}\033[00m" .format(prt))
def prGreen(prt): print("\033[92m {}\033[00m" .format(prt))
def prYellow(prt): print("\033[93m {}\033[00m" .format(prt))
def prLightPurple(prt): print("\033[94m {}\033[00m" .format(prt))
def prPurple(prt): print("\033[95m {}\033[00m" .format(prt))
def prCyan(prt): print("\033[96m {}\033[00m" .format(prt))
def prLightGray(prt): print("\033[97m {}\033[00m" .format(prt))
def prBlack(prt): print("\033[98m {}\033[00m" .format(prt))

def to_numpy(var,use_cuda = False):
    return var.cpu().data.numpy() if use_cuda else var.data.numpy()

def to_tensor(array, use_cuda = False,volatile=False, requires_grad=False):
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        #print('before ndarray is dtype:',ndarray.dtype)
    #ndarray=ndarray.astype('float64')
    #print('after ndarray is dtype:',ndarray.dtype)
    return Variable(
        torch.from_numpy(array), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class HyperParams:
    def __init__(self,hp_dict,is_training):
        self.hidden1=hp_dict['hidden1']
        self.hidden2=hp_dict['hidden2']
        self.rate=hp_dict['rate']
        self.prate=hp_dict['prate']
        self.warmup=hp_dict['warmup']
        self.discount=hp_dict['discount']
        self.bsize=hp_dict['bsize']
        self.rmsize=hp_dict['rmsize']
        self.window_length=hp_dict['window_length']
        self.tau=hp_dict['tau']
        self.ou_theta=hp_dict['ou_theta']
        self.ou_sigma=hp_dict['ou_sigma']
        self.ou_mu=hp_dict['ou_mu']
        self.validate_episodes=hp_dict['validate_episodes']
        self.max_episode_length=hp_dict['max_episode_length']
        self.validate_steps=hp_dict['validate_steps']
        self.output=hp_dict['output']
        self.debug=hp_dict['debug']
        self.init_w=hp_dict['init_w']
        self.train_iter=hp_dict['train_iter']
        self.epsilon=hp_dict['epsilon']
        self.seed=hp_dict['seed']
        self.max_epochs=hp_dict['max_epochs']
        self.is_training=is_training
        self.epsilon_decay=hp_dict['epsilon_decay']
        self.n_frame=hp_dict['n_frame']



