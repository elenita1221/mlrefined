import numpy as np
import math
import time
import matplotlib.pyplot as plt
import copy
import dill  

# autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np  # Thinly-wrapped numpy
import autograd.numpy.random as npr
from autograd.util import flatten
from mlrefined_libraries import superlearn_library as superlib

class learner():
    def __init__(self,**kwargs):    
        # reward structure
        self.reward_struc = kwargs['reward_structure']
        
        # import environment
        self.enviro = kwargs['environment']
        self.num_actions = self.enviro.environment.action_space.n
        
        # container for weight initializations
        self.weight_inits = []
        for i in range(self.num_actions):
            self.weight_inits.append([])
         
        ### initialize q-learning params ###
        # containers for storing various output for analysis
        self.training_episodes_history = []
        self.training_reward = []
        self.validation_reward = []
        self.time_per_episode = []
        
        # container for collecting state / action / reward tuples, recycled every P iterations
        self.gamma = 0.9
        self.max_steps = 1000
        self.exploit_param = 0.5
        self.action_method = 'random'
        self.training_episodes = 100
        self.validation_episodes = 10
        self.normalize_states = False

        # take custom values from args
        if "gamma" in kwargs:
            self.gamma = kwargs['gamma']
        if 'max_steps' in kwargs:
            self.max_steps = kwargs['max_steps']
        if 'action_method' in kwargs:
            self.action_method = kwargs['action_method']
        if 'exploit_param' in kwargs:
            self.exploit = kwargs['exploit_param']
            self.action_method = 'exploit'
        if 'training_episodes' in kwargs:
            self.training_episodes = kwargs['training_episodes']
        if 'validation_episodes' in kwargs:
            self.validation_episodes = kwargs['validation_episodes']
        if 'normalize_states' in kwargs:
            self.normalize_states = kwargs['normalize_states']

    # initialize approximators
    def initialize_approximators(self,**kwargs):
        # grab user arguments
        layer_sizes = kwargs['layer_sizes']
        algo = kwargs['algo']
        self.max_its = kwargs['max_its']
        self.alpha = None
        if 'alpha' in kwargs:
            self.alpha = kwargs['alpha']
        
        # get test point to flush out each approximator
        s_old = self.enviro.environment.reset() 
        x, y, done, info = self.enviro.environment.step(0)       
        x = [x]
        y = [y]
        
        self.approximators = []
        for action in range(self.num_actions):
            # create an instance of the super learner
            func = superlib.superlearner.setup()

            # load in data - need to do this first
            func.load_data(inputs = x, outputs = y)

            # tune dials on desired nonlinearity, cost function, and optimizer
            func.setup_nonlinearity(layer_sizes = layer_sizes ,basis_type = 'neural_net',activation = 'tanh')
           
            #func.setup_nonlinearity(basis_type = 'fixed',activation = 'cos',degree = 3)
            
            func.setup_cost(cost_name = 'least_squares')
            func.setup_optimizer(algo = algo)

            # run the optimizer with '.fit'
            verbose = True
            func.fit(max_its = 1,alpha = 10**-10)     
            
            # empty weight and cost histories
            func.whist.pop()
            func.ghist.pop()
        
            self.approximators.append(func)
     
    # initialize memory characteristics
    def initialize_memory(self,**kwargs):
        # get user args
        self.sample_type = 'consec'
        if 'sample_type' in kwargs:
            self.sample_type = kwargs['sample_type']
            
        self.memory_length = 100
        if 'memory_length' in kwargs:
            self.memory_length = kwargs['memory_length']
            
        self.sample_length = 100
        if 'sample_length' in kwargs:
            self.sample_length = kwargs['sample_length']
            
    
    # evaluate all approximators
    def Q_evals(self,state):
        vals = []
        for action in range(self.num_actions):
            func = self.approximators[action]
            temp = func.predict(state)
            vals.append(temp)
        return vals
    
    # update evaluated points
    def update_evaluations(self):
        # update memory given memory_length
        self.memory = self.memory[-self.memory_length:]
        # what sort of sampling shall we employ?  Consecuitve or random?
        samples = []
        # random subset?
        if self.sample_type == 'random':
            random_inds = np.random.permutation(len(self.memory))
            r = random_inds[:int(self.sample_length)]
            samples = [self.memory[v] for v in r]
            
        # consecutive
        if self.sample_type == 'consec':
            samples = self.memory[:self.sample_length]
        
        # create containers for states and rewards from samples
        self.state_samples = []
        self.reward_samples = []
        for i in range(self.num_actions):
            self.state_samples.append([])
            self.reward_samples.append([])

        # take random samples from memory
        for sample in samples:            
            # extract details from current sample
            s_old = sample[0]
            s_new = sample[1]
            action = sample[2]
            reward = sample[3]
            
            # update reward once approximators online
            if self.num_refreshes >=1 and reward > 0:
                reward += self.gamma*max(self.Q_evals([s_new]))
                reward = reward.tolist()
                reward = reward[0]
                reward = reward[0]
            
            # sort state and reward based on action
            self.state_samples[action].append(s_old)
            self.reward_samples[action].append(reward)

    # loop over approximators and fit
    def update_approximators(self,episode):
        # update approximators with newly sorted data
        for action in range(self.num_actions):
            y = np.asarray(self.reward_samples[action])
            
            # pluck out the approximator
            func = self.approximators[action]
            
            # if any data exists retrain 
            if len(y) > 0:
                y.shape = (len(y),1)
                x = np.asarray(self.state_samples[action])
             
                # run the optimizer with '.fit'
                if self.alpha == None:   # for using line search
                    func.fit(max_its = self.max_its ,inputs = x, outputs = y)
                else:
                    func.fit(max_its = self.max_its ,alpha = self.alpha, inputs = x, outputs = y)
            else:      # if no examples of action taken this round put empty placeholder in weight and cost history for that function approximator
                if len(func.ghist) > 0:
                    func.whist.append(func.whist[-1])
                    func.ghist.append(func.ghist[-1])
                else:
                    func.whist.append([])
                    func.ghist.append([])                    

        # update refresh count and empty current containers
        self.num_refreshes+=1

    ##### Q learning training #####
        # state normalizer
    def state_normalizer(self,state):
        state = state - np.mean(state)
        state = state / np.linalg.norm(state)
        state.shape = (1,len(state))
        return state

    def train(self,**kwargs):     
        self.P = 100     # how often to update model --> in terms of number of episodes
        if 'P' in kwargs:
            self.P = kwargs['P']
        
        self.validate = False
        if 'validate' in kwargs:
            self.validate = kwargs['validate']
            
        num_refresh = 0
        self.memory = []
        self.num_refreshes = 0
        ###### ------ run q-learning ------ ######
        for n in range(self.training_episodes): 
            ### pick this episode's starting position
            # reset arena
            s_old = self.enviro.environment.reset() 
            if self.normalize_states == True:
                s_old = self.state_normalizer(s_old)
                
            # update Q while loc != goal
            episode_history = []      # container for storing this episode's journey
            total_episode_reward = 0
            start = time.clock()

            # loop over episodes, for each run simulation and update Q
            for step in range(self.max_steps):  
                # update episode history container
                episode_history.append(s_old)
      
                #### select action ####
                # select action at random
                action = np.random.randint(self.num_actions)
                
                # select an action greedily with annealing schedule
                random = np.random.rand(1)
                if random > 0.1 and self.num_refreshes >= 1:
                    action = np.argmax(self.Q_evals([s_old]))
                
                # recieve reward, new state, etc., 
                s_new, reward, done, info = self.enviro.environment.step(action)
                if self.normalize_states == True:
                    s_new = self.state_normalizer(s_new)
 
                # check if done, if so adjust reward
                if done:
                    reward = self.reward_struc[-1]
                else: 
                    reward = self.reward_struc[0]
                    
                # add total reward
                total_episode_reward += reward                    
                    
                # collect samples for function approximation
                new_sample = [s_old, s_new, action, reward]
                self.memory.append(new_sample)

                # update old state value
                s_old = s_new
                
                # if pole goes below threshold angle restart - new episode
                if done:
                    self.enviro.environment.reset() 
                    
                    # run validation episodes
                    if self.validate == True:
                        self.run_validation()
                    break
                  
            # run approximator update
            if np.mod(n+1,self.P) == 0:
                self.update_evaluations()
                self.update_approximators(episode = n)
            if np.mod(n,100) == 0:
                print ('episode ' + str(n) + ' complete')
                    
            ### store this episode's computation time and training reward history
            stop = time.clock()
            self.time_per_episode.append(stop - start)
            self.training_episodes_history.append(episode_history)
            self.training_reward.append(total_episode_reward)

        print ('q-learning process complete')
        
    # validate
    def run_validation(self):
        ### store this episode's validation reward history
        ave_reward = 0

        for p in range(self.validation_episodes):
            s_kminus_1 = self.enviro.environment.reset() 
            if self.normalize_states == True:
                s_kminus_1 = self.state_normalizer(s_kminus_1)

            # loop over states
            for t in range(self.max_steps): 
                s_kminus_1.shape = (1,np.size(s_kminus_1))
                                    
                # evluate all action functions on this bias-extended state
                action = np.argmax(self.Q_evals(s_kminus_1))

                # take action, receive output
                s_kminus_1, reward, done, info = self.enviro.environment.step(action)  # reward = +1 for every time unit the pole is above a threshold angle, 0 else
                if self.normalize_states == True:
                    s_kminus_1 = self.state_normalizer(s_kminus_1)

                # exit this episode if complete
                if done:
                    self.enviro.environment.reset() 
                    break
                    
                # record reward
                ave_reward += reward

        ave_reward = ave_reward/float(self.validation_episodes)
        self.validation_reward.append(ave_reward) 