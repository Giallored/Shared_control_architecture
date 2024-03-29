
import torch
from torch.autograd import Variable
import numpy as np



def to_numpy(var,use_cuda = False):
    return var.cpu().data.numpy() if use_cuda else var.data.numpy()

def to_tensor(array, use_cuda = False,volatile=False, requires_grad=False):
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    out = Variable(torch.from_numpy(array), volatile=volatile, requires_grad=requires_grad).type(dtype)
    return out

def to_long_tensor(array, use_cuda = False,volatile=False, requires_grad=False):
    dtype = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    out = Variable(torch.from_numpy(array), volatile=volatile, requires_grad=requires_grad).type(dtype)
    return out


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

def power_decay_schedule(n: int,
                    eps_decay: float,
                    eps_decay_min: float) -> float:
    return max(eps_decay**n, eps_decay_min)

def exponential_annealing_schedule(n, rate):
    return 1 - np.exp(-rate * n)

class HyperParams:
    def __init__(self,hp_dict,is_training):
        self.hidden1=hp_dict['hidden1']
        self.hidden2=hp_dict['hidden2']
        self.rate=hp_dict['rate']
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
        self.max_train_iter=hp_dict['max_train_iter']
        self.sigma=hp_dict['sigma']
        self.seed=hp_dict['seed']
        self.max_epochs=hp_dict['max_epochs']
        self.is_training=is_training
        self.n_frames=hp_dict['n_frames']
        self.epsilon=hp_dict['epsilon']
        self.epsilon_decay=hp_dict['epsilon_decay']


