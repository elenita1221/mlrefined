import matplotlib.animation as animation
from JSAnimation import IPython_display
from IPython.display import clear_output

            
            # close animation of first trials
            self.env.render(close=True)
            
                # animate a single test run
    def animate_test_run(self):
        # start up simulation episode
        observation = self.env.reset()  # note here 'observation' = state

        # start testing phase
        steps = 0
        max_steps = 500
        while steps < max_steps:
            # render action in animation 
            self.env.render()
            
            state = np.insert(observation,0,1)
            state.shape = (len(state),1)

            # pick best action based on input state
            h = np.dot(self.W.T,state)
            action = int(np.argmax(h))

            # take action 
            observation, reward, done, info = self.env.step(action)

            # if pole goes below threshold then end simulation
            if done:
                print("lasted {} timesteps".format(steps))
                break

            steps+=1
        self.env.render(close=True)

class animator():
    ### animate training episode from one version of q-learning ###
    def animate_training_runs(self,**args):  
        
        # make local copies of input
        training_episodes_history = learner.training_episodes_history

        # initialize figure
        fsize = 3
        if grid.width:
            fsize = 5
        fig = plt.figure(figsize = (12,fsize))
        axs = []
        for i in range(len(episodes)):
            ax = fig.add_subplot(1,len(episodes),i+1,aspect = 'equal')
            axs.append(ax)
            
        if len(episodes) == 1:
            axs = np.array(axs)
                
        # compute maximum length of episodes animated
        max_len = 0
        for key in episodes:
            l = len(training_episodes_history[str(key)])
            if l > max_len:
                max_len = l

        # loop over the episode histories and plot the results
        print 'animating run...'
        def show_episode(step):
            # loop over subplots and plot current step of each episode history
            artist = fig

            for k in range(len(axs)):
                ax = axs[k]
                
                # take correct episode
                episode_num = episodes[k]
                current_episode = training_episodes_history[str(episode_num)]
                                
                # define new location of agent
                grid.agent = current_episode[min(step,len(current_episode)-1)]
                
                # color gridworld for this episode and step
                if 'lights' not in args:
                    grid.color_gridworld(ax = ax)
                else:
                    grid.color_gridworld(ax = ax,lights=args['lights'])

                ax.set_title('episode = ' + str(episode_num + 1))
                
            return artist,

        # create animation object
        anim = animation.FuncAnimation(fig, show_episode,frames=min(100,max_len), interval=min(100,max_len), blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = min(100,max_len)/float(10))
        
        print '...done!'
        time.sleep(1)
        clear_output()
        
        return(anim)