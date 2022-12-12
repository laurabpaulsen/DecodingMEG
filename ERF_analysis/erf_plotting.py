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

def get_component_conf(conf, tmin, tmax, sfreq):
    tmin = int(tmin*sfreq)
    tmax = int(tmax*sfreq)
    conf = np.array(conf)
    print(conf.shape)
    for i in range(len(conf)):
        conf_tmp = conf[i]
        #print(conf_tmp)
        for j in range(len(conf)):
            conf[j] = np.mean(conf[i], axis=0)

    return []

def load_data():
    Xbin_load  = np.load(f'../decoding/data/xbin.npz', allow_pickle=True)
    ybin_load = np.load(f'../decoding/data/ybin.npy', allow_pickle=True)
    sessioninds_load = np.load(f'../decoding/data/ybin_seshinds.npy', allow_pickle=True)
       
    Xbin = [Xbin_load[f'arr_{i}'].transpose(1,2,0) for i in range(7)]
    ybin = [ybin_load[i]for i in range(ybin_load.shape[0])]
    sessioninds = [np.array(sessioninds_load[i])for i in range(7)]

    return Xbin, ybin, sessioninds


def plot_var_bins_within_sesh(Xbin, ybin, Xsesh, ysesh , savepath, figsize = (20,10), title = ''):
    ticksize = 12
    ax_titlesize = 18
    header_fontsize = 30
    matplotlib.rc('ytick', labelsize = ticksize)
    matplotlib.rc('xtick', labelsize = ticksize)
    plt.rcParams['font.family'] = 'Times New Roman'

    fig, axs = plt.subplots(7, 3, figsize=figsize, sharey = True, sharex = True, constrained_layout=True)

    means_bin = calculate_means(Xbin, ybin)
    print('means_bin', len(means_bin))
    means_sesh = calculate_means(Xsesh, ysesh)
    print('means_sesh', len(means_sesh))
    conf_bin = calculate_confidence(means_bin)
    conf_sesh = calculate_confidence(means_sesh)

    # plotting
    # variablity across sessions with in same bin
    for i in range(len(Xbin)):
        # plot each bin
        for j in range(len(means_bin)):
            axs[i, 0].plot(means_bin[j], alpha = 0.5, linewidth = 1, label = f'Session {j+1}')

        # plot mean across bins
        #axs[i, 0].plot(np.mean(across_sesh[i], axis = 0), color = 'black', linewidth = 2)
        #axs[i, 0].set_ylabel(f'Bin {i+1}', fontsize = ax_titlesize)
       
        # confidence interval
        #conf = across_sesh_conf[i]
        #axs[i, 0].fill_between(np.arange(250), np.mean(across_sesh[i], axis = 0)-conf, np.mean(across_sesh[i], axis = 0)+conf, color = 'black', alpha = 0.2)
        #axs[i, 2].fill_between(np.arange(250), conf, -conf, alpha = 0.3, label = 'Across session')
        #axs[i, 2].plot(conf, alpha = 0.3)
        #axs[i, 2].plot(-conf, alpha = 0.3)


    # variablity within session across bins
    #for i in range(len(X1)):
        # plot each bin
    #    for j in range(len(within_sesh)):
    #        axs[i, 1].plot(within_sesh[i][j], alpha = 0.5, linewidth = 1, label = f'Bin {j+1}')

        # plot mean across bins
    #    axs[i, 1].plot(np.mean(within_sesh[i], axis = 0), color = 'black', linewidth = 2)
    #    axs[i, 1].set_ylabel(f'Session {i+1}', fontsize = ax_titlesize)
       
        # confidence interval
    #    conf = within_sesh_conf[i]
    #    axs[i, 1].fill_between(np.arange(250), np.mean(within_sesh[i], axis = 0)-conf, np.mean(within_sesh[i], axis = 0)+conf, color = 'black', alpha = 0.2)

        # plot variablity in third column
    #    axs[i, 2].fill_between(np.arange(250), conf, -conf, alpha = 0.3, label = 'Within session')
    #    axs[i, 2].plot(conf, alpha = 0.3)
    #    axs[i, 2].plot(-conf, alpha = 0.3)


    for ax in axs.flatten():
        ax.axhline(0, color = 'black', linewidth = 1, linestyle = '--')
        ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
        ax.set_xlim(0, 250)

    
    for i in range(3):
        axs[-1, i].set_xlabel('Time (ms)', fontsize = ax_titlesize)
        axs[-1, i].set_xlabel('Time (ms)', fontsize = ax_titlesize)
        axs[0, i].set_title(['Across sessions within bin','Within session across bins', '95% confidence interval'][i], fontsize = ax_titlesize)
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

    fig, axs = plt.subplots(1,2, figsize=(20,10), sharey=True)
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
    Xsesh = [X.squeeze() for X in Xsesh[0]]
    plot_var_bins_within_sesh(Xbin, ybin, Xsesh, ysesh, figsize=(15,20), savepath = f'plots/var_bin_sesh_erp_.png', title = f'Difference in ERP between animate and inanimate trials')
    plot_std(Xbin, Xsesh, savepath = f'plots/std_bin_sesh_erf.png')

