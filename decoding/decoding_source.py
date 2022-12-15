"""
Usage: python decoding_source.py
"""

import sys
import decoder_animacy as decoder
import numpy as np

classification = True
ncv = 7
alpha = 0.1
model_type = 'LDA'
get_tgm = True

def load_data(sens = False):
    if not sens:
        Xbin_load  = np.load(f'../subset_data/data/xbin.npz', allow_pickle=True)
        ybin_load = np.load(f'../subset_data/data/ybin.npy', allow_pickle=True)
        sessioninds_load = np.load(f'../subset_data/data/ybin_seshinds.npy', allow_pickle=True)
    else:
        Xbin_load  = np.load(f'../subset_data/data/xbin_sens.npz', allow_pickle=True)
        ybin_load = np.load(f'../subset_data/data/ybin_sens.npy', allow_pickle=True)
        sessioninds_load = np.load(f'../subset_data/data/ybin_seshinds_sens.npy', allow_pickle=True)

    # unpack the loaded data
    Xbin = [Xbin_load[f'arr_{i}']for i in range(7)]
    ybin = [ybin_load[i]for i in range(ybin_load.shape[0])]
    sessioninds = [np.array(sessioninds_load[i])for i in range(7)]
    if not sens:
        Xbin = [X.transpose(1, 2, 0) for X in Xbin]

    return Xbin, ybin, sessioninds

def prep_data(sens = False):
    Xbin, ybin, sessioninds = load_data(sens = sens)
    # create empty lists for each session
    Xsesh = [[] for i in range(7)]
    ysesh = [[] for i in range(7)]

    for i in range(len(Xbin)):
        for j in np.unique(sessioninds[i]):
            sesh_inds = np.where(sessioninds[i] == j)
            X_tmp = Xbin[i][:, sesh_inds, :]
            y_tmp = ybin[i][sesh_inds]

            Xsesh[j].append(X_tmp)
            ysesh[j].append(y_tmp)
       
    return Xbin, ybin, Xsesh, ysesh

def concat_bins(Xsesh, ysesh):
    X = np.concatenate([Xsesh[i].squeeze() for i in range(len(Xsesh))], axis=1)
    y = np.concatenate([ysesh[i] for i in range(len(ysesh))], axis=0)
    return X, y


def run_decoding_leave_bin_out(Xsesh, ysesh, decoder):
    accuracies = [[], [], [], [], [], [], []]
    for session in range(len(Xsesh)):
        for i in range(len(Xsesh[session])):
            bins = Xsesh[session]
            ys = ysesh[session]

            X_test = bins[i].squeeze()
            y_test = ys[i]

            X_train = bins[:]
            X_train.pop(i)
            y_train = ys[:]
            y_train.pop(i)

            X_train, y_train = concat_bins(X_train, y_train)

            acc = decoder.train_test_decoding(X_train, y_train, X_test, y_test)
            print(f'Accuracy for session {session}, without bin {i}: {np.mean(acc)}')
            accuracies[session].append(acc)
    
    return accuracies

def train_test_split_prop(Xsesh, ysesh):
    train_X = [[] for i in range(7)]
    train_y = [[] for i in range(7)]

    test_X = [[] for i in range(7)]
    test_y = [[] for i in range(7)]

    for s in range(len(Xsesh)):
        for b in range(len(Xsesh[s])):
            tmp_X = Xsesh[s][b].squeeze()
            tmp_y = ysesh[s][b].squeeze()

            idx = np.random.choice(np.arange(tmp_X.shape[1]), size = tmp_X.shape[1], replace = False)
            split_idx = np.array_split(idx, 7)
            
            for i, indices in enumerate(split_idx):
                test_tmp_X = tmp_X[:, indices, :]
                test_tmp_y = tmp_y[indices]
                train_inds = np.setdiff1d(idx, indices)
                train_tmp_X = tmp_X[:, train_inds, :]
                train_tmp_y = tmp_y[train_inds]

                # append to train and test
                if b == 0:
                    train_X[s].append(train_tmp_X)
                    train_y[s].append(train_tmp_y)
                    test_X[s].append(test_tmp_X)
                    test_y[s].append(test_tmp_y)
                else:
                    train_X[s][i] = np.concatenate((train_X[s][i], train_tmp_X), axis = 1)
                    train_y[s][i] = np.concatenate((train_y[s][i], train_tmp_y), axis = 0)
                    test_X[s][i] = np.concatenate((test_X[s][i], test_tmp_X), axis = 1)
                    test_y[s][i] = np.concatenate((test_y[s][i], test_tmp_y), axis = 0)
    return train_X, train_y, test_X, test_y


def run_proportional_batch(Xsesh, ysesh, decoder):
    accuracies = [[] for i in range(7)]

    train_X, train_y, test_X, test_y = train_test_split_prop(Xsesh, ysesh)
    for session in range(len(train_X)):
        for i in range(len(train_X[session])):
            acc = decoder.train_test_decoding(train_X[session][i], train_y[session][i], test_X[session][i], test_y[session][i])
            print(f'Accuracy for session {session}: {np.mean(acc)}')
            accuracies[session].append(acc)
        
    return accuracies

if __name__ in '__main__':
    Xbin, ybin, Xsesh, ysesh = prep_data()
    
    decoder = decoder.Decoder(classification=classification, ncv = ncv, alpha = alpha, scale = True, model_type = model_type, get_tgm=get_tgm)

    accuracies = run_proportional_batch(Xsesh, ysesh, decoder)
    np.save(f'./accuracies/accuracies_{model_type}_prop.npy', accuracies)
    
    accuracies = run_decoding_leave_bin_out(Xsesh, ysesh, decoder)
    np.save(f'./accuracies/accuracies_{model_type}_lbo.npy', accuracies)
