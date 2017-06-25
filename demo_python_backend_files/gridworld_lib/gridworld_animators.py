import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from JSAnimation import IPython_display
import time
from IPython.display import clear_output
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

class animator():
        
    ################## animation functions ##################
    ### animate validation runs ###
    def animate_validation_runs(self,**args):
        gridworld = args['gridworld']
        learner = args['learner']
        starting_locations = args['starting_locations']
        
        # make local copies of input
        grid = gridworld
        Q = learner.Q
        starting_locs = starting_locations
        
        # initialize figure
        fsize = 3
        if grid.width:
            fsize = 5
        fig = plt.figure(figsize = (12,fsize))
        axs = []
        for i in range(len(starting_locs)):
            ax = fig.add_subplot(1,len(starting_locs),i+1,aspect = 'equal')
            axs.append(ax)
        
        # only one added subplot, axs must be in array format
        if len(starting_locs) == 1:
            axs = np.array(axs)
        
        ### produce validation runs ###
        print 'animating run...'
        validation_run_history = []
        for i in range(len(starting_locs)):
            # take random starting point - for short just from validation schedule
            grid.agent = starting_locs[i]
            
            # loop over max number of steps and try reach goal
            episode_path = []
            for j in range(grid.max_steps):
                # store current location
                episode_path.append(grid.agent)
                
                ### if you reach the goal end current episode immediately
                if grid.agent == grid.goal:
                    break
                
                # translate current agent location tuple into index
                s_k_1 = grid.state_tuple_to_index(grid.agent)
                    
                # get action
                a_k = grid.get_action(method = 'optimal',Q = Q)
                
                # move based on this action - if move takes you out of gridworld don't move and instead move randomly 
                s_k = grid.get_movin(action = a_k, illegal_move_response = 'random')
  
                # record next step in path
                grid.agent = grid.state_index_to_tuple(state_index = s_k)
                
            # record this episode's path
            validation_run_history.append(episode_path)
        
        ### compute maximum length of episodes animated ###
        max_len = 0
        for i in range(len(starting_locs)):
            l = len(validation_run_history[i])
            if l > max_len:
                max_len = l
        
        ### loop over the episode histories and plot the results ###
        def show_episode(step):
            # loop over subplots and plot current step of each episode history
            artist = fig

            for k in range(len(axs)):
                ax = axs[k]

                # take correct episode
                current_episode = validation_run_history[k]

                # define new location of agent
                loc = current_episode[min(step,len(current_episode)-1)]
                grid.agent = loc

                # color gridworld for this episode and step
                if 'lights' not in args:
                    grid.color_gridworld(ax = ax)
                else:
                    grid.color_gridworld(ax = ax,lights=args['lights'])
                    
                ax.set_title('fully trained run ' + str(k + 1))
                # fig.subplots_adjust(left=0,right=1,bottom=0,top=1)  ## gets rid of the white space around image

            return artist,
        
        # create animation object
        anim = animation.FuncAnimation(fig, show_episode,frames=min(100,max_len), interval=min(100,max_len), blit=True)

        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = min(100,max_len)/float(10))
        
        print '...done!'
        time.sleep(1)
        clear_output()
        
        return(anim)
    
    ### compare training episodes from two q-learning settings ###    
    def animate_training_comparison(self,**args):
        grid = args['gridworld']
        learner_1 = args['learner_1']
        learner_2 = args['learner_2']
        episode = args['episode']
        
        # make local copies of input
        training_episodes_history_v1 = learner_1.training_episodes_history
        training_episodes_history_v2 = learner_2.training_episodes_history
        
        # initialize figure
        fsize = 3
        if grid.width > 10:
            fsize = 5
        fig = plt.figure(figsize = (12,fsize))
        axs = []
        for i in range(2):
            ax = fig.add_subplot(1,2,i+1,aspect = 'equal')
            axs.append(ax)

        # compute maximum length of episodes animated
        max_len = 0
        key = episode
        L1 = len(training_episodes_history_v1[str(key)])
        L2 = len(training_episodes_history_v2[str(key)])
        max_len = max(L1,L2)
        
        # loop over the episode histories and plot the results
        print 'animating run...'
        rewards =  np.zeros((2,1))
        def show_episode(step):
            # loop over subplots and plot current step of each episode history
            artist = fig
            for k in range(len(axs)):
                ax = axs[k]
                
                # take correct episode
                current_episode = 0
                if k == 0:
                    current_episode = training_episodes_history_v1[str(key)]
                else:
                    current_episode = training_episodes_history_v2[str(key)]
                        
                # define new location of agent
                grid.agent = current_episode[min(step,len(current_episode)-1)]
                
                # color gridworld for this episode and step
                if 'lights' not in args:
                    grid.color_gridworld(ax = ax)
                else:
                    grid.color_gridworld(ax = ax,lights=args['lights'])
                                    
                # set title
                if k == 0:
                    ax.set_title('random')
                else:
                    ax.set_title('exploration/exploitation')
            return artist,
        
        # create animation object
        anim = animation.FuncAnimation(fig, show_episode,frames=min(100,max_len), interval=min(100,max_len), blit=True)
        
        # set frames per second in animation
        IPython_display.anim_to_html(anim,fps = min(100,max_len)/float(10))
    
        print '...done!'
        time.sleep(1)
        clear_output()
        
        return(anim)
    
    ### animate training episode from one version of q-learning ###
    def animate_training_runs(self,**args):  
        grid = args['gridworld']
        episodes = args['episodes']
        learner = args['learner']
        
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

    ### draw arrow map after training ###
    ### setup arrows
    def add_arrows(self,ax,state,action):
        x = state[1]
        y = state[0]
        dx = 0
        dy = 0

        ### switch for starting point of arrow depending on action - so that arrow always centered ###
        if action == 0:    # action == down
            y += 0.9
            x += 0.5
            dy = -0.8
        if action == 1:    # action == up
            x += 0.5
            y += 0.1
            dy = 0.8
        if action == 2:    # action == left
            y += 0.5
            x += 0.9
            dx = -0.8
        if action == 3:    # action == right
            y += 0.5
            x += 0.1
            dx = 0.8

        ### add patch with location / orientation determined by action ###
        ax.add_patch(
           patches.FancyArrowPatch(
           (x, y),
           (x+dx, y+dy),
           arrowstyle='->',
           mutation_scale=30,
           lw=2
           )
        )

    # best action map
    def draw_arrow_map(self,world,learner):  
        ### ready state and optimal action lists ###
        # process states for scatter plotting
        states = world.states 

        # process states for scatter plotting
        plot_ready_states = np.zeros((len(states),2))
        for i in range(len(states)):
            a = states[i]
            b = a.split(',')
            state = [int(b[0])]
            state.append(int(b[1]))
            plot_ready_states[i,0] = state[0]
            plot_ready_states[i,1] = state[1]

        ### compute optimal directions ###
        Q = learner.Q
        q_max = np.zeros((len(Q[:,0]),1))
        q_dir = np.zeros((len(Q[:,0]),1))
        for i in range(len(Q[:,0])):
            max_ind = np.argmax(Q[i,:])
            q_dir[i] = max_ind
            max_val = Q[i,max_ind]
            q_max[i] = max_val
        q_dir = q_dir.tolist()
        q_dir = [int(s[0]) for s in q_dir]

        ### plot arrow map ###
        colors = [(0.9,0.9,0.9),(255/float(255), 119/float(255), 119/float(255)), (66/float(255),244/float(255),131/float(255)), (1/float(255),100/float(255),200/float(255)),(0,0,0)]  
        my_cmap = LinearSegmentedColormap.from_list('colormapX', colors, N=100)

        ### setup grid
        p_grid = world.grid
        p_grid[world.goal[0]][world.goal[1]] = 2   
        
        ### setup figrue
        ### setup figure ###
        fig = plt.figure(num=None, figsize = (6,6), dpi=80, facecolor='w', edgecolor='k')

        # plot regression surface 
        ax = plt.subplot(111)
        ax.pcolormesh(p_grid,edgecolors = 'k',linewidth = 0.01,vmin=0,vmax=4,cmap = my_cmap)

        # clean up plot
        ax.set_xlim(-0.1,world.width);
        ax.set_ylim(-0.1,world.height); 

        ### go over states and draw arrows indicating best action
        # for i in range(len(states)):
        for i in range(len(q_dir)):
            state = plot_ready_states[i]
            if state[0] != world.goal[0] or state[1] != world.goal[1]:  
                action = q_dir[i]
                self.add_arrows(ax,state,action)
    
    ### plot Q functions in 3d ###
    def show_Qfunc_in_3d(self,world,learner):
        # process states for scatter plotting
        states = world.states 
        plot_ready_states = np.zeros((len(states),2))
        for i in range(len(states)):
            a = states[i]
            b = a.split(',')
            state = [int(b[0])]
            state.append(int(b[1]))
            plot_ready_states[i,0] = state[0]
            plot_ready_states[i,1] = state[1]
        
        ### plot q functions ###
        # build figure for Q functions
        Q = learner.Q
        labels = ['down','up','left','right']
        colors = ['r','g','b','k']
        fig = plt.figure(num=None, figsize = (10,3), dpi=80, facecolor='w', edgecolor='k')

        ### create figure ###
        for m in range(4):
            # make panel for plotting in 3d
            ax1 = plt.subplot(1,4,m+1,projection = '3d')

            # scatter plot
            ax1.scatter(plot_ready_states[:,0],plot_ready_states[:,1],Q[:,m],c = colors[m])

            # clean up plot
            ax1.view_init(10,20)  
            ax1.set_title('$Q_'+str(m+1) + '$' + ' (' + labels[m] + ')',fontsize = 18)

        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)   # remove whitespace around 3d figure
        plt.show()
     
    ### plot the optimal policy in 3d ###
    def show_optimal_policy_in_3d(self,world,learner):
        # process states for scatter plotting
        states = world.states 
        plot_ready_states = np.zeros((len(states),2))
        for i in range(len(states)):
            a = states[i]
            b = a.split(',')
            state = [int(b[0])]
            state.append(int(b[1]))
            plot_ready_states[i,0] = state[0]
            plot_ready_states[i,1] = state[1]
        
        # compute optimal policy
        Q = learner.Q
        q_max = np.zeros((len(Q[:,0]),1))
        q_dir = np.zeros((len(Q[:,0]),1))
        for i in range(len(Q[:,0])):
            max_ind = np.argmax(Q[i,:])
            q_dir[i] = max_ind
            max_val = Q[i,max_ind]
            q_max[i] = max_val

        q_dir = q_dir.tolist()
        q_dir = [int(s[0]) for s in q_dir]

        #### build optimal policy ####
        # build figure for optimal policy function
        labels = ['down','up','left','right']
        colors = ['r','g','b','k']
        fig = plt.figure(num=None, figsize = (5,5), dpi=80, facecolor='w', edgecolor='k')
       
        ### make panel for plotting in 3d
        ax = plt.subplot(1,1,1,projection = '3d')
        
        # scatter plot
        for i in range(len(plot_ready_states)):
            ax.scatter(plot_ready_states[i,0],plot_ready_states[i,1],q_max[i],c =colors[q_dir[i]])

        # clean up plot
        ax.view_init(40,0)  
        ax.set_title('optimal policy',fontsize = 18)
        ax.legend(labels,loc='center right', bbox_to_anchor=(1, 0.5))
        leg = ax.get_legend()

        for i in range(4):
            leg.legendHandles[i].set_color(colors[i])

        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)   # remove whitespace around 3d figure
        
        ### plot arrow map
        self.draw_arrow_map(world = world,learner = learner)
        plt.show()