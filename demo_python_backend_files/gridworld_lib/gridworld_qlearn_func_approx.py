import numpy as np
import pandas as pd
from autograd import grad as compute_grad   # The only autograd function you may ever need
import copy


class learner():
    def __init__(self,**args):
        # get some crucial parameters from the input gridworld
        self.grid = args['gridworld']
        
        # initialize q-learning params
        self.gamma = 1
        self.max_steps = 5*self.grid.width*self.grid.height
        self.exploit_param = 0.5
        self.action_method = 'exploit'
        self.training_episodes = 500
        self.validation_episodes = 50
        self.training_start_schedule = []
        self.validation_start_schedule = []
        
        # swap out for user defined q-learning params if desired
        if "gamma" in args:
            self.gamma = args['gamma']
        if 'max_steps' in args:
            self.max_steps = args['max_steps']
        if 'action_method' in args:
            self.action_method = args['action_method']
        if 'exploit_param' in args:
            self.exploit = args['exploit_param']
            self.action_method = 'exploit'
        if 'training_episodes' in args:
            self.training_episodes = args['training_episodes']
            # return error if number of training episodes is too big
        if self.training_episodes > self.grid.training_episodes:
            print 'requesting too many training episodes, the maximum num = ' + str(self.grid.training_episodes)
            return 
        self.training_start_schedule = self.grid.training_start_schedule[:self.training_episodes]       
        if 'validation_episodes' in args:
            self.validation_episodes = args['validation_episodes']
            # return error if number of training episodes is too big
        if self.validation_episodes > self.grid.validation_episodes:
            print 'requesting too many validation episodes, the maximum num = ' + str(self.grid.validation_episodes)
            return 
        self.validation_start_schedule = self.grid.validation_start_schedule[:self.validation_episodes]
            
        ##### import function approximators class #####
        # initialize function approximation params and weights
        self.deg = 1
        if 'degree' in args:
            self.deg = args['degree']
            
        self.step_size = 1/float(max(self.grid.height,self.grid.width)*self.deg)*10**-5
        if 'step_size' in args:
            self.step_size = args['step_size']
 
        # switch for choosing various nonlinear approximators
        self.h = 0 
        self.W = 0
        self.num_actions = 4
        if args['approximator'] == 'linear':
            self.h = self.linear_approximator
            
            # initialize weight matrix for function approximator
            self.W = np.random.randn(self.num_actions,self.deg,1 + 2)     # the number of weights per function --> 1 bias, 2 state touching weights (one per state dim)
            
        if args['approximator'] == 'cosine':
            self.h = self.cosine_approximator

            # initialize weight matrix for function approximator
            self.W = np.random.randn(self.num_actions,self.deg,2)     # the number of weights per function --> 1 bias, 2 touching cosine 
        self.W = self.W.astype('float')
            
        # compute gradient of approximator for later use
        self.h_grad = compute_grad(self.h)
        
      
    ##### function approximators #####
    ### -- linear approximator -- ###
    def linear_approximator(self,W):
        # loop over weights and update
        N,M = np.shape(W)
        y = 0
        for n in range(N):
            w = W[n]
            w0 = w[0]
            w1 = w[1:]
            y += w0 + sum(u*v for u,v in zip(w1,self.s))
        return y

    ### -- cosine approximator -- ###
    def cosine_approximator(self,W):                 # Define a function
        N,M = np.shape(W)
        y = 0
        c = 1
        s = np.asarray(self.s)
        for n in range(N):
            w = W[n,:]
            w0 = w[0]
            w1 = w[1]
            y += w0 + w1*np.prod(np.cos((n/float(max(self.grid.height,self.grid.width)))*s)) - 1
        return y
    
    ##### -- function evaluators -- #####
    # evaluate a state through our action-based function approximators
    def evaluate_h(self,s):
        # loop over function approximators and evaluate each one-at-a-time
        self.s = s
        h_eval = []
        for a in range(self.num_actions):
            temp = self.h(self.W[a,:,:])
            h_eval.append(temp)
        return np.asarray(h_eval)
    
    ### Q-learning function - version 1 - take random actions ###
    def train(self,**args):
        # make local (non-deep) copies of globals for easier reading
        grid = self.grid
        gamma = self.gamma
        
        # containers for storing various output
        self.training_episodes_history = {}
        self.training_reward = []
        self.validation_reward = []

        ### start main Q-learning loop ###
        for n in range(self.training_episodes): 
            # set step length
            #self.step_size = 1/float(n+1)
            
            # pick this episode's starting position
            grid.agent = self.training_start_schedule[n]
            
            ### get model functions evaluated at agent current state ###
            # get all function approximator evaluations of current state
            h_eval = self.evaluate_h(grid.agent)
                
            # update Q matrix while loc != goal
            episode_history = []      # container for storing this episode's journey
            total_episode_reward = 0
            for step in range(self.max_steps):   
                # update episode history container
                episode_history.append(grid.agent)
                
                ### if you reach the goal end current episode immediately
                if grid.agent == grid.goal:
                    break
                                   
                ### translate current agent location tuple into index
                s_k_1 = grid.state_tuple_to_index(grid.agent)
                    
                ### get action
                a_k = grid.get_action(method = self.action_method,h = h_eval,exploit_param = self.exploit_param)
                
                #a_k = np.argmax(h_eval)
                
                # compute previous approximator value
                h = copy.deepcopy(h_eval)
                h_s_k_1 = h[a_k]
                
                ### move based on this action
                s_k = grid.get_movin(action = a_k)
                
                ### update current location of agent 
                grid.agent = grid.state_index_to_tuple(state_index = s_k)
                
                ### get reward     
                r_k = grid.get_reward(state_index = s_k) 
                
                ### update model params ###
                # transform current state using chosen function approximator
                h_eval = self.evaluate_h(grid.agent)
                
                # update Q function data
                ind = np.argmax(h_eval)
                h_k = h_eval[ind]
                q_k = r_k + gamma*h_k
                
                # update model given new datapoint
                grad = self.h_grad(self.W[a_k,:,:])   
                self.W[a_k,:,:] = self.W[a_k,:,:] - self.step_size*(h_s_k_1 - q_k)*grad    
                
                # update training reward
                total_episode_reward+=r_k
                
            # print out update if verbose set to True
            if 'verbose' in args:
                if args['verbose'] == True:
                    if np.mod(n+1,50) == 0:
                        print 'training episode ' + str(n+1) +  ' of ' + str(self.training_episodes) + ' complete'
            
            ### store this episode's training reward history
            self.training_episodes_history[str(n)] = episode_history
            self.training_reward.append(total_episode_reward)
            
            
        print 'q-learning algorithm complete'
   
