import math
import numpy as np
import random
from functools import reduce
import operator
from RL_agent.utils import exponential_annealing_schedule


class UniformReplayBuffer():
    # uniform sampling
    
    # the buffer is a collection of vectors representing flattened transition
    def __init__(self, capacity: int,   # the capacity of the buffer
                       o_shape:  tuple,
                       s_shape:  tuple, # the shape of a state
                       a_shape=(1,)):   # the shape of an action
        self.o_shape = o_shape
        self.s_shape = s_shape
        self.a_shape = a_shape
        
        self.o_size = reduce(operator.mul, o_shape, 1)
        self.s_size = reduce(operator.mul, s_shape, 1) # how many numbers to store a state
        self.a_size = reduce(operator.mul, a_shape, 1) # how many numbers to store an action

        # the size of a transition: o+s+a+r+d+s'+o' 
        self.t_size = self.o_size + self.s_size + self.a_size + 1 + 1 + self.o_size + self.s_size 

        self.capacity = capacity
        self.idx = -1
        
        self.buffer = {"observation":  np.zeros((capacity,*o_shape),dtype=np.float32),
                       "state":      np.zeros((capacity,*s_shape),dtype=np.float32),
                       "action":     np.zeros((capacity,*a_shape),dtype=np.float32),
                       "reward":     np.zeros(capacity,dtype=np.float32),
                       "done":       np.zeros(capacity,dtype=np.float32),
                       "next_observation":  np.zeros((capacity,*o_shape),dtype=np.float32),
                       "next_state": np.zeros((capacity,*s_shape),dtype=np.float32),
}
        
        
        self.full = False # wether the buffer is full or it contains empty spots

    def size(self):
        if self.full:
            return self.capacity
        else:
            return self.idx+1

    def store(self, o: np.ndarray,
                    s: np.ndarray, 
                    a: np.ndarray, # or int if discrete actions 
                    r: float, 
                    d: bool, 
                    op: np.ndarray,
                    sp: np.ndarray):
        
        self.idx += 1
        if (self.idx >= self.capacity):
            if not self.full:
                self.full = True # the buffer is full
                print("buffer full")
            self.idx = 0     # reset the index (start to overwrite old experiences)
        self.buffer["observation"][self.idx,...] = o.copy()
        self.buffer["state"][self.idx,...] = s.copy()
        self.buffer["action"][self.idx] = a if type(a) == int else a.copy()
        self.buffer["reward"][self.idx] = r
        self.buffer["done"][self.idx] = d
        self.buffer["next_observation"][self.idx,...] = op.copy()
        self.buffer["next_state"][self.idx,...] = sp.copy()



    def sample_idxes_weights(self,n):
        high = self.size()
        return random.choices(population=range(high), k=n), None     

    def get_capacity(self):
        if self.cnt < self.capacity:
            return self.idx / self.capacity
        else:
            return 1.0
        
    def sample(self, n: int):
        # random.sample performs sampling without replacement
        idxes, w = self.sample_idxes_weights(n)


        # model wants states to be in shape [n, s_shape]
        observations = self.buffer["observation"][idxes]
        states =     self.buffer["state"][idxes]
        actions =    self.buffer["action"][idxes]
        rewards =    self.buffer["reward"][idxes]
        dones =      self.buffer["done"][idxes]
        next_observations = self.buffer["next_observation"][idxes]
        new_states = self.buffer["next_state"][idxes]


        return (observations,states,actions,rewards,dones,next_observations,new_states,idxes,w)
    

class PrioritizedReplayBuffer(UniformReplayBuffer):

    def __init__(self, capacity: int, o_shape:tuple,s_shape: tuple, a_shape=(1, ), alpha=0.6, beta_0= 1e-2, beta_inc=1.001):
        super().__init__(capacity,o_shape, s_shape, a_shape)
        if math.ceil(math.log2(capacity)) != math.floor(math.log2(capacity)):
            capacity = 2**math.ceil(math.log2(capacity))
            print(f"rescaling buffer to the next power of two: {capacity}.")
        
        # store the priorities in a tree
        self.priorities = SumTree(capacity)
        self.max_priority = 1.0
        self.cnt=1

        self.alpha = alpha
        self.beta = beta_0
        self.beta_inc = beta_inc
        self.beta_aneling =lambda n: exponential_annealing_schedule(n,rate=1e-4)


    def sample_idxes_weights(self, n):
        high = self.size()

        (idxes, Ps) = self.priorities.sample_batch(n)

        w = (high*Ps)**-self.beta

        w /= w.max()
        if self.beta < 1: # beta annealing
             self.beta = self.beta_aneling(self.cnt)
        return idxes, w

    def store(self, o: np.ndarray, 
                    s: np.ndarray, 
                    a: np.ndarray, # or int if discrete actions 
                    r: float, 
                    d: bool, 
                    op: np.ndarray, 
                    sp: np.ndarray,):
        super().store(o,s,a,r,d,op,sp)
        self.cnt+=1
        self.priorities.set_priority(self.idx,self.max_priority)

    def update_priorities(self, idxes, td_errors, eps=1e-6):
        updated_priorities = np.abs(td_errors)**self.alpha + eps

        _m = updated_priorities.max()
        if _m > self.max_priority: # update the maximum priority
            self.max_priority = _m

        for i in range(len(idxes)):
            self.priorities.set_priority(idxes[i],updated_priorities[i])
    




class SumTree():
    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.size = 2*n_bins - 1
        self.data = np.zeros(self.size)
        self.height = math.log2(n_bins)

    def _left(self, i):
        return 2*i+1

    def _right(self, i):
        return 2*i+2

    def _parent(self, i):
        return (i-1) // 2

    def _update_cumulative(self, i):
        value_left = self.data[self._left(i)]
        value_right = self.data[self._right(i)]
        self.data[i] = value_left + value_right

        if i == 0: # the root of the tree
            return
        else: # update the parent
            self._update_cumulative(self._parent(i)) 

    def _is_leaf(self, i):
        # it is a leaf if it's stored in the last self.n_bins positions
        return i >= self.size - self.n_bins 

    def _importance_sampling(self, priority, i=0):
        # https://adventuresinmachinelearning.com/sumtree-introduction-python/
        if self._is_leaf(i):
            # return transition to which i corresponds
            return i - (self.size - self.n_bins), self.data[i] 
        else:
            value_left = self.data[self._left(i)]
            # value_right = self.data[self._right(i)]
            
            if priority < value_left:
                return self._importance_sampling(priority, self._left(i))
            else: # priority >= value_left
                return self._importance_sampling(priority-value_left, self._right(i))

    def get_sum(self):
        return self.data[0]        

    def set_priority(self, idx, priority):
        # where is the leaf stored on the array
        pos = self.size - self.n_bins + idx

        self.data[pos] = priority
        self._update_cumulative(self._parent(pos))

    def sample_batch(self, k):
        rng = self.get_sum() / k
        # low variance sampling like in particle filter
        unif = np.random.uniform() * rng
        
        idxes = np.zeros(k, dtype=np.uint32)
        Ps = np.zeros(k)

        for i in range(k):
            idxes[i], Ps[i]  = self._importance_sampling(unif)
            unif += rng
        return idxes, Ps