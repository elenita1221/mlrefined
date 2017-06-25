import gym
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import copy

# autograd functionality
from autograd import grad as compute_grad   # The only autograd function you may ever need
import autograd.numpy as np  # Thinly-wrapped numpy
import autograd.numpy.random as npr
from autograd.util import flatten

class learner():
    def __init__(self,**args):    
        # import environment
        self.enviro = args['environment']
       
        # load in start schedule for training / validation
        self.training_start_schedule = self.enviro.training_start_schedule
        self.validation_start_schedule = self.enviro.validation_start_schedule
                
        ### setup network arhitecture for each action function ###
        self.layer_sizes = [np.shape(self.enviro.environment.observation_space)[0]+1,5, 1]
        
        # initialize parameters of network
        if 'layer_sizes' in args:
            self.layer_sizes = args['layer_sizes']
            
        # produce weights for each action function, store in dictionary
        self.num_actions = self.enviro.environment.action_space.n
        self.all_weights = {}
        for a in range(self.num_actions):
            # initialize each function's weights randomly
            init_scale = 0.1
            weights = self.init_random_params(scale = init_scale, layer_sizes=self.layer_sizes)
            self.all_weights[a] = weights
        
        # create gradient function of network 
        self.h_grad = compute_grad(self.h_function)
        
        # initialize global representative of state to plug in to each action function
        self.state = 0

    ##### function approximator functions #####
    # initialize parameters
    def init_random_params(self,scale, layer_sizes, rs=npr.RandomState(0)):
        """Build a list of (weights, biases) tuples, one for each layer."""
        return [(rs.randn(insize, outsize) * scale,   # weight matrix
                 rs.randn(outsize) * scale)           # bias vector
                for insize, outsize in zip(self.layer_sizes[:-1], self.layer_sizes[1:])]

    # run through  
    def nn_predict(self,weights, inputs, nonlinearity=np.cos):
        for W, b in weights:
            outputs = np.dot(inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs
    
    # evaluate an action function
    def h_function(self,weights):
        h_eval = self.nn_predict(weights,self.state)
        return h_eval
        
    ##### Q learning training #####
    def train(self,**args):
        ### initialize q-learning params ###
        self.gamma = 1
        self.max_steps = 1000
        self.exploit_param = 0.5
        self.action_method = 'random'
        self.training_episodes = 100
        self.alpha = 10**-4

        # take custom values from args
        if "alpha" in args:
            self.alpha = args['alpha']
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
        if self.training_episodes > self.enviro.training_episodes:
            print 'requesting too many training episodes, the maximum num = ' + str(self.enviro.training_episodes)
            return     
        
        # make a few local copies of things to keep the algorithm visually clean
        gamma = self.gamma
        alpha = self.alpha
        
        # containers for storing various output
        self.training_episodes_history = {}
        self.training_reward = []
        self.validation_reward = []
        self.time_per_episode = []
        
        ###### ------ run q-learning ------ ######
        ### re-initialize all action-function weights ###
        self.num_actions = self.enviro.environment.action_space.n
        self.all_weights = {}
        for a in range(self.num_actions):
            # initialize each function's weights randomly
            init_scale = 0.1
            weights = self.init_random_params(scale = init_scale, layer_sizes=self.layer_sizes)
            self.all_weights[a] = weights

        for n in range(self.training_episodes): 
            ### pick this episode's starting position
            # reset arena
            observation = self.enviro.environment.reset() 
            
            # pick next starting state from pre-made schedule
            observation = self.training_start_schedule[n]

            # extend the given state by adding 1 for a bias
            self.state = np.insert(1,0,observation)
            
            # evluate all action functions on this bias-extended state
            h = []
            for a in range(self.num_actions):
                weights = self.all_weights[a]
                h_a = self.h_function(weights)
                h.append(h_a)
                           
            # update Q while loc != goal
            episode_history = []      # container for storing this episode's journey
            total_episode_reward = 0
            start = time.clock()

            # loop over episodes, for each run simulation and update Q
            for step in range(self.max_steps):   
                # update episode history container
                episode_history.append(observation)
      
                #### select action ####
                # select action at random
                action = self.enviro.environment.action_space.sample()
                
                # select greedy with certain probability
                if self.action_method == 'exploit':
                    r = np.random.rand(1)
                    if r < self.exploit_param:
                        action = int(np.argmax(h))
                    
                # recieve reward, new state, etc., 
                observation, reward, done, info = self.enviro.environment.step(action)  # reward = +1 for every time unit the pole is above a threshold angle, 0 else
                
                # extend the given state by adding 1 for a bias
                self.state = np.insert(1,0,observation)
                
                # update total reward
                if done:
                    reward = -10
                total_episode_reward += reward

                ### update model weights ###
                # get previous state evaluation
                h_i = h[action]
            
                # evluate all action functions on this bias-extended state
                h = []
                for a in range(self.num_actions):
                    weights = self.all_weights[a]
                    h_a = self.h_function(weights)
                    h.append(h_a)
 
                # compute new q value and update constant
                h_max = np.max(h)
                q_k = reward + gamma*h_max
                u_k = self.alpha*(h_i - q_k)
                
                # update proper function's weights
                action_weights = self.all_weights[action]
                new_weights = []
                gradient = self.h_grad(action_weights)
                for i in range(len(action_weights)):
                    temp1 = action_weights[i][0] - u_k*gradient[i][0]
                    temp2 = action_weights[i][1] - u_k*gradient[i][1]
                    temp3 = (temp1,temp2)
                    new_weights.append(temp3)
                action_weights = new_weights
                self.all_weights[action] = action_weights
            
                # if pole goes below threshold angle restart - new episode
                if done:
                    self.enviro.environment.reset() 
                    break
                    
            ### store this episode's computation time and training reward history
            stop = time.clock()
            self.time_per_episode.append(stop - start)
            self.training_episodes_history[str(n)] = episode_history
            self.training_reward.append(total_episode_reward)
            
            ### store this episode's validation reward history
            if 'validate' in args:
                if args['validate'] == True:
                    ave_reward = 0
                    for p in range(len(self.validation_start_schedule)):
                        observation = self.validation_start_schedule[p]
                        
                        # loop over states
                        for t in range(self.max_steps): 
                            # use optimal policy to choose action
                            self.state = np.insert(observation,0,1)
                            
                            # evluate all action functions on this bias-extended state
                            h = []
                            for a in range(self.num_actions):
                                weights = self.all_weights[a]
                                h_a = self.h_function(weights)
                                h.append(h_a)
 
                            action = int(np.argmax(h))
                            
                            # take action, receive output
                            observation, reward, done, info = self.enviro.environment.step(action)  # reward = +1 for every time unit the pole is above a threshold angle, 0 else
                            
                            # record reward
                            ave_reward += reward
                            
                            # exit this episode if complete
                            if done:
                                self.enviro.environment.reset() 
                                break
                            
                    ave_reward = ave_reward/float(len(self.validation_start_schedule))
                    self.validation_reward.append(ave_reward)


        print 'q-learning process complete'

  