"""
Usage: python decoding_across_session.py
"""

import sys
import decoder_animacy as decoder
import numpy as np
from decoding_source import prep_data, concat_bins
sys.path.append('../ERF_analysis')
from prep_data import balance_class_weights

classification = True
ncv = 10
alpha = 0.5
model_type = 'LDA'
get_tgm = True

# idea
# analogous to leave batch out = leave one session out for testing on
# analogous to proportional batch = same proportion of data from each session

def leave_session_out(Xsesh, ysesh, decoder):
    # note: Should there be cross validation within the test and train set?
    accuracies = [[], [], [], [], [], [], []]
    for i in range(len(Xsesh)):
        Xtest, ytest = concat_bins (Xsesh[i], ysesh[i])

        Xtrain = []
        ytrain = []
        for j in range(len(Xsesh)):
            if j != i:
                X_tmp, y_tmp = concat_bins(Xsesh[j], ysesh[j])
                Xtrain.append(X_tmp)
                ytrain.append(y_tmp)
        Xtrain, ytrain = concat_bins(Xtrain, ytrain)
        acc = decoder.run_decoding_across_sessions(Xtrain, ytrain, Xtest, ytest)
        accuracies[i] = acc
        print(f'Finished session {i} of {len(Xsesh)}')

    return accuracies

def split_session(X_list, y_list):
    """
    splits the data of a session into 7 parts
    """
    X, y = concat_bins(X_list, y_list)

    idx = np.arange(X.shape[1])
    np.random.shuffle(idx)
    idx_split = np.array_split(idx, 7)

    X_split = []
    y_split = []
    for i in range(len(idx_split)):
        X_split.append(X[:, idx_split[i], :])
        y_split.append(y[idx_split[i]])
    return X_split, y_split

def proportional_session(Xsesh, ysesh, decoder):
    accuracies = [[], [], [], [], [], [], []]
    Xtrain = []
    ytrain = []
    Xtest = []
    ytest = []
    for s in range(len(Xsesh)):
        X_tmp, y_tmp = split_session(Xsesh[s], ysesh[s])
        for j in range(len(X_tmp)):
            X_tmp_pop = X_tmp.pop(j)
            y_tmp_pop = y_tmp.pop(j)
            if s == 0:
                Xtest.append(X_tmp_pop)
                ytest.append(y_tmp_pop)
            else:
                Xtest[j] = np.concatenate((Xtest[j], X_tmp_pop), axis=1)
                ytest[j] = np.concatenate((ytest[j], y_tmp_pop), axis=0)

            X_tmp_train, y_tmp_train = concat_bins(X_tmp, y_tmp)
            X_tmp.insert(j, X_tmp_pop)
            y_tmp.insert(j, y_tmp_pop)

            if s == 0:
                Xtrain.append(X_tmp_train)
                ytrain.append(y_tmp_train)
            else:
                Xtrain[j] = np.concatenate((Xtrain[j], X_tmp_train), axis=1)
                ytrain[j] = np.concatenate((ytrain[j], y_tmp_train), axis=0)

    for i in range(7):
        acc = decoder.run_decoding_across_sessions(Xtrain[i], ytrain[i], Xtest[i], ytest[i])
        accuracies[i].append(acc)
        print(f'Finished {i} of 7')
        
    return accuracies

if __name__ in '__main__':
    Xbin, ybin, Xsesh, ysesh = prep_data()
    decoder = decoder.Decoder(classification=classification, ncv = 1, alpha = alpha, scale = True, model_type = model_type, get_tgm=get_tgm)
    accuracies = leave_session_out(Xsesh, ysesh, decoder)
    np.save(f'./accuracies/accuracies_{model_type}_lso.npy', accuracies)

    accuracies = proportional_session(Xsesh, ysesh, decoder)
    np.save(f'./accuracies/accuracies_{model_type}_props.npy', accuracies)