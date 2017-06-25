import gym

class environment():
    def __init__(self,**args):
        # grab enviro arg
        self.environment_name = args['environment']
        self.environment = 0
            
        # initialize enviroment
        #### cartpole ####
        '''
        Note here that there are four features to the given state:
        - the cart position -> the 1st entry of the state
        - the cart velocity -> the 2nd entry of the state
        - the angle of the pole measured as its deviation from the vertical position - 3rd entry of state (in radians)
        - the angular velocity of the pole - 4th entry of state
        '''
        if self.environment_name == 'cartpole':
            self.environment = gym.make('CartPole-v0') 

        #### create starting schedule of states ####
        # create training episodes
        self.training_episodes = 500
        if 'training_episodes' in args:
            # define num of training episodes
            self.training_episodes = args['training_episodes']

        # make new training start schedule
        self.training_start_schedule = self.make_start_schedule(episodes = self.training_episodes)
        
        # preset number of training episodes value
        self.validation_episodes = 100
        if 'validation_episodes' in args:
            # define num of testing episodes
            self.validation_episodes = args['validation_episodes']
            
        # make new testing start schedule
        self.validation_start_schedule = self.make_start_schedule(episodes = self.validation_episodes)
                
    ### create starting schedule - starting position of each episode of training or testing ###
    def make_start_schedule(self,**args):
        num_episodes = args['episodes']
        start_schedule = []
        
        # create schedule of random starting positions for each episode
        for i in range(num_episodes):
            # get random initial state from env
            loc = self.environment.reset() 
            start_schedule.append(loc)

        return start_schedule
