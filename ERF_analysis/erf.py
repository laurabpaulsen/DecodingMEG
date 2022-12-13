import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from mycolorpy import colorlist as mcp
import sys
sys.path.append('../decoding')
from decoding_source import prep_data


def get_std(X):
    stds = []
    for i in range(len(X)):
        # average across labels
        all_mean = np.mean(X[i], axis = 2)

        # standard deviation across trials of the difference
        std = np.std(all_mean, axis = 1)
        stds.append(std)
    return stds

def calculate_means(X, y):
    means = [[], [], [],[],[],[],[]]
    
    for a in range(len(X)): # loop over bins
        X_tmp = X[a]
        y_tmp = y[a]

        animate_inds = (y_tmp == 1)
        animate_X = X_tmp[:, animate_inds, :]
        inanimate_X = X_tmp[:, ~animate_inds, :]
        animate_mean = np.mean(animate_X, axis = (1,2))
        inanimate_mean = np.mean(inanimate_X, axis = (1,2))

        means[a].append(animate_mean-inanimate_mean)
    
    return means



def calculate_confidence(list_of_means):
    conf = [[],[],[],[],[],[],[]]
    for i in range(len(list_of_means)):
        for j in range(len(list_of_means[i])):
            conf[i].append(np.std(np.array(list_of_means[i][j]), axis=1)/np.sqrt(list_of_means[i][j].shape[1])*1.96)
    
    print('len conf', len(conf))
    return conf



def plot_var_bins_within_sesh(Xbin, ybin, Xsesh, ysesh , savepath, figsize = (20,10), title = ''):
    ticksize = 12
    ax_titlesize = 18
    header_fontsize = 30
    alpha_ci = 0.3 # for confidence interval
    matplotlib.rc('ytick', labelsize = ticksize)
    matplotlib.rc('xtick', labelsize = ticksize)
    plt.rcParams['font.family'] = 'Times New Roman'

    colour_ani = '#0063B2FF'
    colour_inani = '#5DBB63FF'

    fig, axs = plt.subplots(7, 2, figsize=figsize, dpi = 300, sharey = True, sharex = True, constrained_layout=True)

    means_bin = calculate_means(Xbin, ybin)
    means_sesh = calculate_means(Xsesh, ysesh)
    for i in range(len(means_bin)):
        axs[i, 0].plot(means_bin[i][0], color = 'black', linewidth = 1, label = 'Difference')
        axs[i, 1].plot(means_sesh[i][0], color = 'black', linewidth = 1, label = 'Difference')

    
    for i in range(len(Xbin)):
        X_tmp = Xbin[i]
        y_tmp = ybin[i]
        animate_inds = (y_tmp == 1)
        animate_X = X_tmp[:, animate_inds, :]
        inanimate_X = X_tmp[:, ~animate_inds, :]
        animate_mean = np.mean(animate_X, axis = (2))
        inanimate_mean = np.mean(inanimate_X, axis = (2))
        axs[i, 0].plot(np.mean(animate_mean, axis = 1), color = colour_ani, linewidth = 1,  label = 'Animate')
        axs[i, 0].plot(np.mean(inanimate_mean, axis = 1), color = colour_inani, linewidth = 1, label = 'Inanimate')

        # confidence interval
        axs[i, 0].fill_between(np.arange(250), np.mean(animate_mean, axis = 1)+np.std(animate_mean, axis = 1)/np.sqrt(animate_mean.shape[1])*1.96, np.mean(animate_mean, axis = 1)-np.std(animate_mean, axis = 1)/np.sqrt(animate_mean.shape[1])*1.96, alpha = alpha_ci, color = colour_ani)
        axs[i, 0].fill_between(np.arange(250), np.mean(inanimate_mean, axis = 1)+np.std(inanimate_mean, axis = 1)/np.sqrt(inanimate_mean.shape[1])*1.96, np.mean(inanimate_mean, axis = 1)-np.std(inanimate_mean, axis = 1)/np.sqrt(inanimate_mean.shape[1])*1.96, alpha = alpha_ci, color = colour_inani)
    
    for i in range(len(Xbin)):
        X_tmp = Xsesh[i]
        y_tmp = ysesh[i]
        animate_inds = (y_tmp == 1)
        animate_X = X_tmp[:, animate_inds, :]
        inanimate_X = X_tmp[:, ~animate_inds, :]
        animate_mean = np.mean(animate_X, axis = (2))
        inanimate_mean = np.mean(inanimate_X, axis = (2))
        axs[i, 1].plot(np.mean(animate_mean, axis = 1), color = colour_ani, linewidth = 1, label = 'Animate')
        axs[i, 1].plot(np.mean(inanimate_mean, axis = 1), color = colour_inani, linewidth = 1, label = 'Inanimate')

        # confidence interval
        axs[i, 1].fill_between(np.arange(250), np.mean(animate_mean, axis = 1)+np.std(animate_mean, axis = 1)/np.sqrt(animate_mean.shape[1])*1.96, np.mean(animate_mean, axis = 1)-np.std(animate_mean, axis = 1)/np.sqrt(animate_mean.shape[1])*1.96, alpha = alpha_ci, color = colour_ani)
        axs[i, 1].fill_between(np.arange(250), np.mean(inanimate_mean, axis = 1)+np.std(inanimate_mean, axis = 1)/np.sqrt(inanimate_mean.shape[1])*1.96, np.mean(inanimate_mean, axis = 1)-np.std(inanimate_mean, axis = 1)/np.sqrt(inanimate_mean.shape[1])*1.96, alpha = alpha_ci, color = colour_inani)
        

    for ax in axs.flatten():
        ax.axhline(0, color = 'black', linewidth = 1, linestyle = '--')
        ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        ax.set_xlim(0, 250)

    
    for i in range(2):
        axs[-1, i].set_xlabel('Time (ms)', fontsize = ax_titlesize)
        axs[-1, i].set_xlabel('Time (ms)', fontsize = ax_titlesize)
        axs[0, i].set_title(['Across sessions within bin','Within session across bins'][i], fontsize = ax_titlesize)
        axs[0, i].legend(fontsize = ticksize, loc = 'upper right')
        #axs[7, i].legend(fontsize = ticksize, loc = 'upper right')
        #axs[7, i].set_ylabel(['All bins','All sessions', ''][i], fontsize = ax_titlesize)
    

    fig.supylabel('Animate inanimate difference', fontsize = ax_titlesize)
    fig.suptitle(title, fontsize = header_fontsize)
    plt.savefig(savepath)
    plt.show()


def plot_std(Xbin, Xsesh, savepath = None):
    std_bins = get_std(Xbin)
    std_session = get_std(Xsesh)
    np.save('std_sessions.npy', std_session)

    fig, axs = plt.subplots(1,2, figsize=(20,10), dpi = 300, sharey=True)
    for ax in axs:
        if ax == axs[0]:
            std = std_session
            ax.set_title('Within session', fontsize=20)
            for i in range(len(std)):
                ax.plot(std[i], label=f'Session {i+1}', alpha=0.5)
            ax.plot(np.mean(np.array(std), axis = 0), color='black', linewidth=3, label='Mean')
            #ax.fill_between(np.arange(0, 250), np.mean(np.array(std), axis = 0) - np.std(np.array(std), axis = 0), np.mean(np.array(std), axis = 0) + np.std(np.array(std), axis = 0), color='black', alpha=0.2)

        else:
            std = std_bins
            ax.set_title('Across session', fontsize=20)
            for i in range(len(std)):
                ax.plot(std[i], label=f'Bin {i+1}', alpha=0.5)
            ax.plot(np.mean(np.array(std), axis = 0), color='black', linewidth=3, label='Mean')
            #ax.fill_between(np.arange(0, 250), np.mean(np.array(std), axis = 0) - np.std(np.array(std), axis = 0), np.mean(np.array(std), axis = 0) + np.std(np.array(std), axis = 0), color='black', alpha=0.2)
            
        ax.legend(loc = 'upper right')

    for a in axs.flatten():
        a.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        a.set_xlim(0, 250)

    # ylim
    #axs[0].set_ylim(0, 0.1)
    axs[0].set_ylabel('Standard deviation across trials', fontsize=16)

    if savepath != None:
        plt.savefig(savepath)


if __name__ == '__main__':
    Xbin, ybin, Xsesh, ysesh = prep_data()
    Xsesh = [np.concatenate(i, axis = 2) for i in Xsesh]
    Xsesh = [np.transpose(i.squeeze(), (0,1,2)) for i in Xsesh]
    ysesh = [np.concatenate(i, axis = 0) for i in ysesh]

    plot_var_bins_within_sesh(Xbin, ybin, Xsesh, ysesh, figsize=(15,20), savepath = f'plots/bin_sesh_erp_animate_vs_inanimate.png', title = f'Difference in ERF between animate and inanimate trials')
    plot_std(Xbin, Xsesh, savepath = f'plots/std_bin_sesh_erf.png')

