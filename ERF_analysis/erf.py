import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
sys.path.append('../decoding')
from decoding import prep_data

# set font for all plots
plt.rcParams['font.family'] = 'times new roman'
plt.rcParams['image.cmap'] = 'RdBu_r' # note: delete everywhere else
plt.rcParams['image.interpolation'] = 'bilinear'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300


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



def plot_var_bins_within_sesh(Xsesh, ysesh, figsize, savepath, title = ''):

    alpha_ci = 0.2 # for confidence interval

    colour_ani = '#0063B2FF'
    colour_inani = '#5DBB63FF'

    fig, axs = plt.subplots(2, 4, figsize=figsize, dpi = 300, sharey = True, sharex = True)
    means_sesh = calculate_means(Xsesh, ysesh)

    for i, ax in enumerate(axs.flatten()):
        if i < 7:
            X_tmp = Xsesh[i]
            y_tmp = ysesh[i]
            animate_inds = (y_tmp == 1)
            animate_X = X_tmp[:, animate_inds, :]
            inanimate_X = X_tmp[:, ~animate_inds, :]
            animate_mean = np.mean(animate_X, axis = (2))
            inanimate_mean = np.mean(inanimate_X, axis = (2))
            ax.plot(np.mean(animate_mean, axis = 1), color = colour_ani, linewidth = 1, label = 'Animate')
            ax.plot(np.mean(inanimate_mean, axis = 1), color = colour_inani, linewidth = 1, label = 'Inanimate')
            ax.plot(means_sesh[i][0], color = 'black', linewidth = 1, label = 'Difference')

            # confidence interval
            ax.fill_between(np.arange(250), np.mean(animate_mean, axis = 1)+np.std(animate_mean, axis = 1)/np.sqrt(animate_mean.shape[1])*1.96, np.mean(animate_mean, axis = 1)-np.std(animate_mean, axis = 1)/np.sqrt(animate_mean.shape[1])*1.96, alpha = alpha_ci, color = colour_ani)
            ax.fill_between(np.arange(250), np.mean(inanimate_mean, axis = 1)+np.std(inanimate_mean, axis = 1)/np.sqrt(inanimate_mean.shape[1])*1.96, np.mean(inanimate_mean, axis = 1)-np.std(inanimate_mean, axis = 1)/np.sqrt(inanimate_mean.shape[1])*1.96, alpha = alpha_ci, color = colour_inani)
            ax.axhline(0, color = 'black', linewidth = 1, linestyle = '--')
            ax.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
            ax.set_xlim(0, 250)

            # title
            ax.set_title('Session ' + str(i+1))

    fig.supxlabel('Time (ms)')

    # plot legend on the last subplot
    # get handles from the first subplot
    handles, labels = axs[0, 0].get_legend_handles_labels()

    # plot legend on the last subplot
    axs[-1, -1].legend(handles, labels, loc = 'center', fontsize = 12)
    axs[-1, -1].axis('off')


    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(savepath)


def plot_std(X, savepath = None, blocks = False, ymin = 0, ymax = 0.5, mean = True, sens = False):
    # check if sens is either 'grad' or 'mag' or False
    if sens not in ['grad', 'mag', False]:
        raise ValueError('sens must be either "grad" or "mag" or False')
    if sens == 'grad':
        # take the first 204 channels
        X = [x[:, :, :204] for x in X]
    elif sens == 'mag':
        # take the last channels
        X = [x[:, :, 204:] for x in X]
    
    std = get_std(X)

    fig, axs = plt.subplots(1,1, figsize=(7,4), dpi = 300, sharey=True)

    for i in range(len(std)):
        if blocks:
            label = f'Block {i+1}'
        else:
            label = f'Session {i+1}'
        axs.plot(std[i], label=label, alpha=0.5)
    
    if mean:
        mean_std = np.mean(std, axis=0)
        axs.plot(mean_std, label='Mean', color='black', linewidth=2)

    axs.legend(loc = 'upper right')

    axs.set_xticks(np.arange(0, 251, step=50), [0. , 0.2, 0.4, 0.6, 0.8, 1. ])
    axs.set_xlim(0, 250)

    axs.set_ylabel('Standard deviation')
    axs.set_ylim(ymin, ymax)


    if savepath != None:
        plt.savefig(savepath)

    plt.close()


if __name__ == '__main__':
    # Source space
    Xbin, ybin, Xsesh, ysesh = prep_data()

    Xsesh = [np.concatenate(i, axis = 2) for i in Xsesh]
    Xsesh = [i.squeeze() for i in Xsesh]
    ysesh = [np.concatenate(i, axis = 0) for i in ysesh]

    plot_var_bins_within_sesh(Xsesh, ysesh, figsize=(10,7), savepath = f'plots/sesh_erp_animate_vs_inanimate.png')
    plot_std(Xbin, savepath = f'plots/std_block_source.png', blocks=True, ymin = 0.025, ymax = 0.04)
    plot_std(Xsesh, savepath = f'plots/std_sesh_source.png', ymin = 0.025, ymax = 0.04)

    # Sensor space
    Xbin, ybin, Xsesh, ysesh = prep_data(sens = True)

    Xsesh = [np.concatenate(i, axis = 2) for i in Xsesh]
    Xsesh = [i.squeeze() for i in Xsesh]
    ysesh = [np.concatenate(i, axis = 0) for i in ysesh]
    plot_std(Xbin, savepath=f'plots/std_block_sens_grad.png', ymin = 0.00000000000025, ymax = 0.0000000000006, blocks=True, sens = 'grad')
    plot_std(Xbin, savepath=f'plots/std_block_sens_mag.png', ymin = 0.00000000000035, ymax = 0.0000000000012, blocks=True, sens = 'mag')
    plot_std(Xsesh, savepath=f'plots/std_sesh_sens_grad.png', ymin = 0.00000000000025, ymax = 0.0000000000006, sens = 'grad')
    plot_std(Xsesh, savepath=f'plots/std_sesh_sens_mag.png', ymin = 0.00000000000035, ymax = 0.0000000000012, sens = 'mag')

