"""
Creates plots of the decoding results.

Usage: python decoding_plots.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from decoding import prep_data

# set font for all plots
plt.rcParams['font.family'] = 'times new roman'
plt.rcParams['image.cmap'] = 'RdBu_r' # note: delete everywhere else

def chance_level():
    Xbin, ybin, Xsesh, ysesh = prep_data()
    del Xbin, ybin, ysesh
    Xsesh = [np.concatenate(i, axis = 2) for i in Xsesh]
    Xsesh = [np.transpose(i.squeeze(), (0,1,2)) for i in Xsesh]
    
    n_trials = [i.shape[1] for i in Xsesh]
    chance_level = []
    for i in range(len(n_trials)):
        n, p= n_trials[i], 0.5
        # get the chance level at alpha = 0.05
        k = binom.ppf(0.95, n, p)
        chance_level.append(k/n)
    return chance_level

def plot_tgm_diagonal(lbo, prop, savepath = None):
    """
    Plot temporal generalization matrix and diagonal for each session.
    """
    vmin = 0.30
    vmax = 0.70

    # filling out a page 
    cm = 1/2.54  # centimeters in inches
    figsize = (18*cm, 24*cm)
    fig, ax = plt.subplots(len(lbo), 4, figsize = figsize, dpi = 400, sharex=True, sharey='col')
    for i, (session_lbo, session_prop) in enumerate(zip(lbo, prop)):
        # compute the average of each session
        avg_lbo = np.mean(session_lbo, axis = 0)
        avg_prop = np.mean(session_prop, axis = 0)

        # plot the average image for each session
        ax[i, 0].imshow(avg_lbo, vmin = vmin, vmax = vmax, cmap = 'RdBu_r')
        ax[i, 0].set_ylabel(f'Session {i+1}', fontsize = 8)
        ax[i, 0].set_yticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])

        ax[i, 2].imshow(avg_prop, vmin = vmin, vmax = vmax, cmap = 'RdBu_r')
        ax[i, 2].set_yticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])


        for j in range(len(session_prop)):
            ax[i, 3].plot(session_prop[j].diagonal(), linewidth = 0.3, alpha = 0.5)
            ax[i, 1].plot(session_lbo[j].diagonal(), linewidth = 0.3, alpha = 0.5)

        # plot the average diagonal for each session
        ax[i, 1].plot(avg_lbo.diagonal(), color = 'k', linewidth = 0.5)
        ax[i, 3].plot(avg_prop.diagonal(), color = 'k', linewidth = 0.5)


    ax[1, 0].set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    ax[1, 0].set_xlim(0, 250)

    ax[0, 0].set_title('Temporal Generalization')
    ax[0, 0].set_title('Leave Block Out')
    ax[0, 2].set_title('Proportional Batch')
    #ax[0, 1].legend(loc = 'upper right')
    ax[0, 1].set_ylim(0.3, 0.8)
    ax[0, 3].set_ylim(0.3, 0.8)

    # size of tick labels
    for a in ax.flatten():
        a.tick_params(axis='both', which='major', labelsize=6)
    fig.supxlabel('Time (s)', fontsize = 8)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)

def plot_diagonal(lbo, prop, savepath = None):
    """
    Plot the diagonal for each session and the confidence interval.
    """
    # filling out a page 
    cm = 1/2.54  # centimeters in inches
    figsize = (18*cm, 24*cm)
    fig, ax = plt.subplots(len(lbo), 2, figsize = figsize, dpi = 400, sharex=True, sharey=True)

    for i, (session_lbo, session_prop) in enumerate(zip(lbo, prop)):
        avg_lbo = np.mean(session_lbo, axis = 0)
        avg_prop = np.mean(session_prop, axis = 0)
        ax[i,0].fill_between(range(len(session_lbo[0])), avg_lbo.diagonal() - 1.96*np.std(avg_lbo, axis = 0)/np.sqrt(len(lbo[i])), avg_lbo.diagonal() + 1.96*np.std(avg_lbo, axis = 0)/np.sqrt(len(lbo[i])), color = 'grey', alpha = 0.4)
        ax[i,1].fill_between(range(len(session_prop[0])), avg_prop.diagonal() - 1.96*np.std(avg_prop, axis = 0)/np.sqrt(len(prop[i])), avg_prop.diagonal() + 1.96*np.std(avg_prop, axis = 0)/np.sqrt(len(prop[i])), color = 'grey', alpha = 0.4)
        
        ax[i,0].plot(avg_lbo.diagonal(), color = 'darkblue', linewidth = 1)
        ax[i,1].plot(avg_prop.diagonal(), color = 'darkblue', linewidth = 1)

    # set the x axis to go from 0 to 1000 ms
    for a in ax.flatten():
        a.set_xticks(np.arange(0, 1001, step=200), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        a.set_xlim(0, 250)

    if savepath is not None:
        plt.savefig(savepath)

def plot_tgm_sesh(lso, props, savepath = None):
    vmin = 0.30
    vmax = 0.70
    fig, ax = plt.subplots(1, 2, figsize = (8, 4), dpi = 300, sharey='row', sharex = True)

    avg_lso = np.mean(lso, axis = 0)
    ax[0].imshow(avg_lso, vmin = vmin, vmax = vmax, cmap = 'RdBu_r', origin = 'lower')
    ax[0].set_title('Leave Session Out')

    avg_prop = np.mean(props, axis = 0)
    ax[1].imshow(avg_prop, vmin = vmin, vmax = vmax, cmap = 'RdBu_r', origin = 'lower')
    ax[1].set_title('Proportional Session')

    ax[0].set_yticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    ax[0].set_ylim(0, 250)


    for a in ax.flatten():
        a.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        a.set_xlim(0, 250)
    
    # times the y ticks by 100 
    ## get current y ticks

    fig.supxlabel('Time (s)')
    fig.supylabel('Time (s)')

    #plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)

def plot_diagonal_sesh(lso, props, savepath: None):
    fig, ax = plt.subplots(1, 2, figsize = (8, 4), dpi = 300, sharey='row', sharex = True) 

    avg_lso = np.mean(lso, axis = 0)
    ax[0].set_title('Leave Session Out')
    avg_prop = np.mean(props, axis = 0)
    ax[1].set_title('Proportional Session')
    for j in range(len(lso)):
        ax[0].plot(lso[j].diagonal(), linewidth = 0.5, alpha = 0.5, label = f'{j+1}')
        ax[1].plot(props[j].diagonal(), linewidth = 0.5, alpha = 0.5, label = f'{j+1}')
   
    ax[0].plot(avg_lso.diagonal(), color = 'k', linewidth = 1)
    ax[1].plot(avg_prop.diagonal(), color = 'k', linewidth = 1)
    ax[0].legend(loc = 'upper right', title = 'Testing on \n session', title_fontsize = 6, fontsize = 6)
    ax[1].legend(loc = 'upper right', title = 'Fold', title_fontsize = 6, fontsize = 6)

    for a in ax.flatten():
        a.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        a.set_xlim(0, 250)
    
    #fig.suptitle('Across Session Decoding', fontsize = 14)
    fig.supxlabel('Time (s)')
    fig.supylabel('Decoding accuracy')

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)

def tgm_cross(cross, savepath = None):

    cm = 1/2.54  # centimeters in inches
    figsize = (18*cm, 18*cm)

    vmin = 0.35
    vmax = 0.65

    fig, axs = plt.subplots(cross.shape[0], cross.shape[1], figsize = figsize, dpi = 300, sharey=True, sharex = True)

    for i in range(cross.shape[0]):
        for j in range(cross.shape[1]):
            axs[i, j].imshow(cross[i, j], vmin = vmin, vmax = vmax, cmap = 'RdBu_r', origin = 'lower')
            axs[i, j].set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ], fontsize = 6)
            axs[i, j].set_yticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ], fontsize = 6)
            # rotate x ticks a bit
            for tick in axs[i, j].get_xticklabels():
                tick.set_rotation(90)

    for j in range(cross.shape[1]):
        axs[j, 0].set_ylabel(f'Session {j+1}', fontsize = 6)
        axs[0, j].set_title(f'Session {j+1}', fontsize = 6)
    fig.supylabel('Testing session', fontsize = 6)
    fig.suptitle('Training session', fontsize = 6)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)

def diagonal_cross(cross, savepath = None):
    vmin = 0.40
    vmax = 0.70
    cm = 1/2.54  # centimeters in inches
    figsize = (18*cm, 7*cm)
    fig, ax = plt.subplots(2, 4, figsize = figsize, dpi = 500)
    for i, a in enumerate(ax.flatten()):
        if i < 7:
            for j in range(cross.shape[1]):
                if j == 0 and i == 0:
                    a.axhline(y = 0.5333333333333333, color = 'k', linewidth = 0.3, linestyle = '--', alpha = 0.4, label = 'Chance')
                else:
                    a.axhline(y = 0.5333333333333333, color = 'k', linewidth = 0.3, linestyle = '--', alpha = 0.4)
                a.plot(cross[i, j].diagonal(), linewidth = 0.4, alpha = 0.7, label = f'Session {j+1}')
                a.set_title(f'Training on session {i+1}', fontsize = 8)
                a.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ], fontsize = 6)
                a.set_ylim(vmin, vmax)
                # change fontsize of y ticks
                a.tick_params(axis='y', labelsize=6)

    
    # get the legend labels from the first axis and plot them on the last axis
    handles, labels = ax[0, 0].get_legend_handles_labels()
    ax[-1, -1].axis('off')
    ax[-1, -1].legend(handles, labels, loc = 'center', title = 'Testing', title_fontsize = 6, fontsize = 6)
    #ax[-1, -1].legend(loc = 'upper right', title = 'Testing on \n session', title_fontsize = 6, fontsize = 6)

    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath)

def plot_tgm_difference(a1, a2, savepath = None):
    vmin = -0.025
    vmax = 0.025

    if a1.shape and a2.shape != (7, 7, 250, 250):
        raise ValueError('Input arrays must be of shape (7, 7, 250, 250)')
    
    mean_a1 = np.mean(a1, axis = (0, 1))
    mean_a2 = np.mean(a2, axis = (0, 1))

    fig, axs = plt.subplots(1, 1, figsize = (7, 7), dpi = 300)

    im = axs.imshow(mean_a1 - mean_a2, vmin = vmin, vmax = vmax, cmap = 'RdBu_r', origin = 'lower')
    # show the colorbar on the right in a smaller size
    plt.colorbar(im, ax = axs, shrink = 0.5, pad = 0.05)
    # add title over the colorbar
    plt.title('Sensor space minus source space accuracy', fontsize = 16)


    if savepath is not None:
        plt.savefig(savepath)



if __name__ in '__main__':
    lbo = np.load('./accuracies/accuracies_LDA_lbo.npy', allow_pickle=True) # leave batch out
    propb = np.load('./accuracies/accuracies_LDA_prop.npy', allow_pickle=True) # proportional batch
    cross = np.load('./accuracies/cross_decoding_ncv_5.npy', allow_pickle=True).squeeze() # cross session
    cross_sens = np.load('./accuracies/cross_decoding_sens_ncv_5.npy', allow_pickle=True).squeeze() # cross session

    plot_tgm_diagonal(lbo, propb,  savepath = f'./plots/tgm_diagonal_within_session.png')
    plot_diagonal(lbo, propb, savepath = f'./plots/diagonal_within_session.png')

    tgm_cross(cross, savepath = f'./plots/cross_session_tgm.png')
    tgm_cross(cross_sens, savepath = f'./plots/cross_session_tgm_sens.png')
    
    diagonal_cross(cross, savepath = f'./plots/cross_session_diagonal.png')
    diagonal_cross(cross_sens, savepath = f'./plots/cross_session_diagonal_sens.png')

    plot_tgm_difference(cross_sens, cross, savepath = f'./plots/cross_session_tgm_difference.png')
