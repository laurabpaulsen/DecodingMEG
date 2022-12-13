import matplotlib.pyplot as plt
import numpy as np


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

        # plot the 95 % confidence interval for the lbo session
        #ax[i, 1].fill_between(range(len(session_lbo[0])), avg_lbo.diagonal() - 1.96*np.std(avg_lbo, axis = 0)/np.sqrt(len(lbo[i])), avg_lbo.diagonal() + 1.96*np.std(avg_lbo, axis = 0)/np.sqrt(len(lbo[i])), color = 'k', alpha = 0.2)
        #ax[i, 1].axhline(0.5, color = 'k', linestyle = '--', linewidth = 0.5)

        # plot the 95 % confidence interval for the prop session
        #ax[i, 3].fill_between(range(len(session_prop[0])), avg_prop.diagonal() - 1.96*np.std(avg_prop, axis = 0)/np.sqrt(len(prop[i])), avg_prop.diagonal() + 1.96*np.std(avg_prop, axis = 0)/np.sqrt(len(prop[i])), color = 'k', alpha = 0.4)
        #ax[i, 3].axhline(0.5, color = 'k', linestyle = '--', linewidth = 0.5)

    ax[1, 0].set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    ax[1, 0].set_xlim(0, 250)

    ax[0, 0].set_title('Temporal Generalization')
    ax[0, 0].set_title('Leave Batch Out')
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
    fig.supxlabel('Training session', fontsize = 6)


    if savepath is not None:
        plt.savefig(savepath)


if __name__ in '__main__':
    lbo = np.load('./accuracies/accuracies_LDA_lbo.npy', allow_pickle=True) # leave batch out
    propb = np.load('./accuracies/accuracies_LDA_prop.npy', allow_pickle=True) # proportional batch
    lso = np.load('./accuracies/accuracies_LDA_lso.npy', allow_pickle=True) # leave session out
    props = np.load('./accuracies/accuracies_LDA_props.npy', allow_pickle=True).squeeze() # proportional session

    cross = np.load('./accuracies/cross_decoding_13-12-2022-20-01.npy', allow_pickle=True).squeeze() # cross session
    

    plot_tgm_diagonal(lbo, propb,  savepath = f'./plots/tgm_diagonal_within_session.png')
    plot_diagonal(lbo, propb, savepath = f'./plots/diagonal_within_session.png')

    plot_tgm_sesh(lso, props, savepath = f'./plots/tgm_between_session.png')
    plot_diagonal_sesh(lso, props, savepath = f'./plots/diagonal_between_session.png')

    tgm_cross(cross, savepath = f'./plots/cross_session_tgm.png')