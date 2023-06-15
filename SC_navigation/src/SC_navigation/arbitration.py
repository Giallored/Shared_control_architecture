from RL_agent import DDPG



class Arbitation:

    def __init__(self,n_states, n_actions, args):
        
        self.agent=DDPG(n_states,n_actions,args)
