from collections import namedtuple


class Dataset:

    def __init__(self,name:str,max_size:int,verbose =False):
        self.dict = {}
        self.name = name
        self.max_size=max_size
        self.cur_size=0
        self.capacity=0.
        self.verbose=verbose
        self.Buffer = namedtuple('Sample',field_names=['step','goal', 'observation','action'])

    def append(self,step,goal,observation,action):
        if self.capacity<=1.0:
            self.dict[step]=self.Buffer(step,goal,observation,action)
            self.cur_size+=1

            self.capacity = self.cur_size/self.max_size
        
    def is_full(self):
        if self.capacity<=1.0:
            if self.verbose: print(f"Buffer is at {round(self.capacity*100) }% of capacity", end="\r", flush=True)
            return True
        else:
            if self.verbose: print(f"Buffer is full!", end="\r", flush=True)
            return False
        
    
    def save(self,output):
        pass



