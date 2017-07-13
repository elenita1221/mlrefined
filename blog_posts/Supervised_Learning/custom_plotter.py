import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import gridspec

# custom plot for spiffing up plot of a single mathematical function
def single_plot(table):
    # is the function 2-d or 3-d?
    dim = np.shape(table)[1]
    
    # single two dimensonal plot
    if dim == 2:   
        # plot the line
        plt.style.use('ggplot')
        fig = plt.figure(figsize = (12,4))

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,2, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # plot input function
        ax2 = plt.subplot(gs[1])
        ax2.plot(table[:,0], table[:,1], c='r', linewidth=2)

        # plot x and y axes, and clean up
        ax2.grid(True, which='both')
        ax2.axhline(y=0, color='k', linewidth=1)
        ax2.axvline(x=0, color='k', linewidth=1)
        ax2.set_xlabel('$x$',fontsize = 15)
        ax2.set_ylabel('$y$',fontsize = 15,rotation = 0)
        plt.show()
        
    # single 3-d function plot
    if dim == 3:    
        # plot the line
        plt.style.use('ggplot')
        fig = plt.figure(figsize = (15,6))

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,2, 1]) 
        ax1 = plt.subplot(gs[0]); ax1.axis('off');
        ax3 = plt.subplot(gs[2]); ax3.axis('off');

        # plot input function
        ax2 = plt.subplot(gs[1],projection='3d')
        ax2.plot_surface(table[:,0], table[:,1], table[:,2], alpha = 0.3,color = 'r',rstride=10, cstride=10,linewidth=2,edgecolor = 'k')

        # plot x and y axes, and clean up
        ax2.set_xlabel('$x_1$',fontsize = 20)
        ax2.set_ylabel('$x_2$',fontsize = 20,rotation = 0)
        ax2.set_zlabel('$y$',fontsize = 20)

        # clean up plot and set viewing angle
        ax2.view_init(10,30)
        plt.show()
        
        
# custom plot for spiffing up plot of a two mathematical functions
def double_plot(table1,table2,**kwargs): 
    # plot the functions 
    fig = plt.figure(figsize = (15,4))
    plt.style.use('ggplot')
    ax1 = fig.add_subplot(121); ax2 = fig.add_subplot(122); 
    plot_type = 'scatter'
    if 'plot_type' in kwargs:
        plot_type = kwargs['plot_type']
    if plot_type == 'scatter':
        ax1.scatter(table1[:,0], table1[:,1], c='r', s=20)
        ax2.scatter(table2[:,0], table2[:,1], c='r', s=20)
    if plot_type == 'continuous':
        ax1.plot(table1[:,0], table1[:,1], c='r', linewidth=2)
        ax2.plot(table2[:,0], table2[:,1], c='r', linewidth=2)

    # plot x and y axes, and clean up
    ax1.grid(True, which='both'), ax2.grid(True, which='both')
    ax1.axhline(y=0, color='k', linewidth=1), ax2.axhline(y=0, color='k', linewidth=1)
    ax1.axvline(x=0, color='k', linewidth=1), ax2.axvline(x=0, color='k', linewidth=1)
    plt.show()
    