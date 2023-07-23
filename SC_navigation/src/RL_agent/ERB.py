from collections import namedtuple, deque
import operator
import random
import numpy as np


class Prioritized_ERB:

    def __init__(self, n_frames,memory_size=10000):
        
        self.items_cnt=0            # Counter to identify the samples     
        self.memory_size=memory_size
        self.alpha = 0.5
        self.alpha_decay_rate = 0.99
        self.beta = 0.5
        self.beta_growth_rate = 1.001
        self.n_frames=3
        # In the replay buffer you store replays 
        self.replay = namedtuple("Replay",field_names=["observation", "varState", "action", "reward", "done", "next_observation", "next_varState"])

        # In the data buffer you store the data associated with replays
        self.data = namedtuple("Data", field_names=["priority", "probability", "weight", "index"])
                                
        indexes = []
        datas=[]
        for i in range(memory_size):
            indexes.append(i)
            d = self.data(0,0,0,i)
            datas.append(d)
    
        self.replay_buffer = {key: self.replay for key in indexes}
        self.data_buffer = {key: data for key, data in zip(indexes, datas)}

        self.alpha_sum_p = 0
        self.p_max = 1
        self.w_max = 1
    
    def get_capacity(self):
        if self.items_cnt < self.memory_size:
            return  self.items_cnt / self.memory_size
        else:
            return 1.0

    
    def store(self, state ,  action ,  reward ,  done ,  next_state ):
        observation = state[0]
        varState = state[1]
        next_observation =next_state[0]
        next_varState = next_state[1]
        index = self.items_cnt % self.memory_size
        self.items_cnt+=1

        if self.items_cnt == self.memory_size:
          print('---------------BUFFER SATURATED-----------------')

        # CASE: Buffer is full
        if self.items_cnt > self.memory_size:
            # Get the index of the "oldest" item in the buffer
            old_item = self.data_buffer[index]
            self.alpha_sum_p -= old_item.priority**self.alpha
            
            # removed the max ==> need to update every max_ variable
            if old_item.priority == self.p_max:
                self.data_buffer[index].priority = 0
                self.p_max = max(self.data_buffer.items(), key=operator.itemgetter(1)).priority
            
            if old_item.weight == self.w_max:
                self.data_buffer[index].weight = 0
                self.w_max = max(self.data_buffer.items(), key=operator.itemgetter(2)).weight
        
        # new samples are instantiated with max priority and weight (since are totatlly new experience)
        priority = self.p_max
        weight = self.w_max

        # Get the probability
        self.alpha_sum_p += priority ** self.alpha
        probability = priority ** self.alpha / self.alpha_sum_p
        
        # new samples for both buffers
        exp = self.replay(observation, varState, action, reward, done, next_observation, next_varState)
        self.replay_buffer[index] = exp
        data = self.data(priority, probability, weight, index)
        self.data_buffer[index] = data


    def sample(self,batch_size):
        
        # Sample the batach from the memory, weighting the choice according to the probabilities 
        list_data = list(self.data_buffer.values())
        batch = random.choices(self.data_buffer,[val.probability for val in list_data],k=batch_size)

        # Get indices and weights associated
        indices = tuple([item.index for item in batch])
        weights = tuple([item.weight for item in batch])
        
        obs0_batch=[]
        sVar0_batch=[]
        action_batch=[]
        terminal1_batch=[]
        reward_batch=[]
        obs1_batch =[]
        sVar1_batch=[]
        
        # Sample the buffer using the indices got before
        for i in indices:
            item =self.replay_buffer.get(i)
            obs0_batch.append(item.observation)
            sVar0_batch.append(item.varState)
            action_batch.append(item.action)
            terminal1_batch.append(item.done)
            reward_batch.append(item.reward)
            obs1_batch.append(item.next_observation)
            sVar1_batch.append(item.next_varState)
        
        # Prepare and validate parameters.
        obs0_batch = np.array(obs0_batch).reshape(batch_size,self.n_frames,-1)
        obs1_batch = np.array(obs1_batch).reshape(batch_size,self.n_frames,-1)
        sVar0_batch = np.array(sVar0_batch).reshape(batch_size,-1)
        sVar1_batch = np.array(sVar1_batch).reshape(batch_size,-1)
        terminal1_batch = np.array(terminal1_batch).reshape(batch_size,-1)
        reward_batch = np.array(reward_batch).reshape(batch_size,-1)
        action_batch = np.array(action_batch).reshape(batch_size,-1)

        

        return  obs0_batch,sVar0_batch,action_batch,reward_batch,terminal1_batch, obs1_batch,sVar1_batch,indices,weights


    # update 
    def update_data(self, TDs, indices):
      
        for td, index in zip(TDs, indices):

            # Get the N
            N = min(self.items_cnt, self.memory_size)
            
            # get the new priority as TD error and the new weights
            priority_new = td
            weight_new = ((N * priority_new)**(-self.beta))/self.w_max
            
            # update the MAXes 
            if priority_new > self.p_max:
                self.p_max = priority_new
            if weight_new > self.w_max:
                self.w_max = weight_new
            
            # Sample the old priority from the memory
            priority_old = self.data_buffer[index].priority
            
            #Compute the new probability and store it back in memory
            self.alpha_sum_p -= priority_old**self.alpha
            self.alpha_sum_p += priority_new**self.alpha
            probability_new = td**self.alpha / self.alpha_sum_p

            try:
                self.data_buffer[index] = self.data(priority_new, probability_new, weight_new, index) 
            except:
                self.data_buffer[index] = self.data(priority_new, probability_new, 1, index) 

    #update of all the parameters
    def update_param(self):
        self.alpha *= self.alpha_decay_rate
        self.beta *= self.beta_growth_rate
        if self.beta > 1:
            self.beta = 1

        N = min(self.items_cnt, self.memory_size)

        self.alpha_sum_p = 0
        for element in self.data_buffer.values():
            self.alpha_sum_p += element.priority**self.alpha
        
        for element in self.data_buffer.values():
            if element.priority == 0:       # if the item has never been instantiated
                pass
            else:
                probability = element.priority**self.alpha / self.alpha_sum_p
                weight = ((N *  element.probability)**(-self.beta))/self.w_max
                d = self.data(element.priority, probability, weight, element.index)
                self.data_buffer[element.index] = d
