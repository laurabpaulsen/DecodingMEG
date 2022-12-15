
import numpy as np
import mne
import json

def balance_class_weights(X, y):
    keys, counts = np.unique(y, return_counts = True)
    if counts[0]-counts[1] > 0:
        index_inanimate = np.where(np.array(y) == 0)
        random_choices = np.random.choice(len(index_inanimate[0]), size = counts[0]-counts[1], replace=False)
        remove_ind = [index_inanimate[0][i] for i in random_choices]
    else:
        index_animate = np.where(np.array(y) == 1)
        random_choices = np.random.choice(len(index_animate[0]), size = counts[1]-counts[0], replace=False)
        remove_ind = [index_animate[0][i] for i in random_choices]

    X_equal = np.delete(X, remove_ind, axis = 1)
    y_equal = np.delete(y, remove_ind, axis = 0)

    print(f'Removed a total of {len(remove_ind)} trials')
    print(f'{len(y_equal)} remains')
    
    return X_equal, y_equal, remove_ind

def read_and_concate_sessions(session_files, trigger_list):
    """Reads and concatenates epochs from different sessions into a single epochs object.
    Parameters
    ----------
    session_files : list
        List of session files to be concatenated.
    trigger_list : list
        List of triggers to be included in the concatenated epochs object.
    
    Returns
    -------
    X : concatenated trials from sessions
    y : concatenated labels from sessions
    """
    for i in session_files:
        epochs = mne.read_epochs(f'/media/8.1/final_data/laurap/epochs/{i}-epo.fif')
        if i == session_files[0]:
            y = epochs.events[:, 2]
            idx = [i for i, x in enumerate(y) if x in trigger_list]
            X = np.load(f'/media/8.1/final_data/laurap/source_space/parcelled/{i}_parcelled.npy', allow_pickle = True)
            X = X.transpose(2, 0, 1)
            X = X[:, idx, :]
            y = np.array(y)[idx]
        else:
            y_tmp = epochs.events[:, 2]
            idx = [i for i, x in enumerate(y_tmp) if x in trigger_list]
            X_tmp = np.load(f'/media/8.1/final_data/laurap/source_space/parcelled/{i}_parcelled.npy', allow_pickle = True)
            X_tmp = X_tmp.transpose(2, 0, 1)
            X_tmp = X_tmp[:, idx, :]
            y_tmp = np.array(y_tmp)[idx]
            # before we concatenate, the correlation of each label is checked
            # if the correlation is negative, the sign is flipped
            for k in range(X_tmp.shape[2]):
                # take means over trials
                mean1 = np.mean(X[:, :, k], axis = 1)
                mean2 = np.mean(X_tmp[:, :, k], axis = 1)

                # take correlation
                corr = np.corrcoef(mean1, mean2)[0, 1]
                if corr < 0:
                    X_tmp[:, :, k] = X_tmp[:, :, k] * -1
                    print(f'Flipped sign of label {k} in session {i} because correlation was negative')
                    
            X = np.concatenate((X, X_tmp), axis = 1)
            y = np.concatenate((y, y_tmp), axis = 0)

    return X, y

def create_blocks(sessions, n_bins, n, animate_triggers):
    session_inds = []
    for i,session in enumerate(sessions):
        X = sessions[i][0]
        y = sessions[i][1]

        y = [1 if i in animate_triggers else 0 for i in y]

        n_trials_sesh  = int(len(y))
        n_trials_bin  = int(n_trials_sesh/n_bins)

        min = n_trials_bin*n
        max = n_trials_bin*(n+1)

        if session == sessions[0]:
            X_block = X[:, min:max, :]
            y_block = y[min:max]
            session_inds.extend([i]*len(y_block))

        else:    
            X_block_tmp = X[:, min:max, :]
            y_block_tmp = y[min:max]
            session_inds.extend([i]*len(y_block_tmp))
            X_block = np.concatenate((X_block, X_block_tmp), axis = 1)
            y_block = np.concatenate((y_block, y_block_tmp))
        
    X_block, y_block, remove_ind = balance_class_weights(X_block, y_block)
    
    if remove_ind != []:
        session_inds = np.delete(np.array(session_inds), np.array(remove_ind), axis = 0)

    return X_block.transpose(2,0,1), y_block, session_inds



def main():
    with open('../event_ids.txt', 'r') as f:
        # reading in file with information about the triggers
        file = f.read()
        event_ids = json.loads(file)

    animate_triggers = [value for key, value in event_ids.items() if 'Animate' in key]
    inanimate_triggers = [value for key, value in event_ids.items() if 'Inanimate' in key][:len(animate_triggers)]
    triggers = animate_triggers.copy()
    triggers.extend(inanimate_triggers)

    trig = [key for key, value in event_ids.items() if value in triggers]

    sessions = [['visual_03', 'visual_04'], ['visual_05', 'visual_06', 'visual_07'], ['visual_08', 'visual_09', 'visual_10'], ['visual_11', 'visual_12', 'visual_13'],['visual_14', 'visual_15', 'visual_16', 'visual_17', 'visual_18', 'visual_19'],['visual_23', 'visual_24', 'visual_25', 'visual_26', 'visual_27', 'visual_28', 'visual_29'],['visual_30', 'visual_31', 'visual_32', 'visual_33', 'visual_34', 'visual_35', 'visual_36', 'visual_37', 'visual_38']]
    sessions_data = [read_and_concate_sessions(i, triggers) for i in sessions]

    # sign flipping for each session
    for i in range(len(sessions_data)):
        if i != 0:
            X = sessions_data[i][0]
            y = sessions_data[i][1]

            for k in range(X.shape[2]):
                # take means over trials
                mean1 = np.mean(sessions_data[0][0][:, :, k], axis = 1)
                mean2 = np.mean(X[:, :, k], axis = 1)

                # take correlation
                corr = np.corrcoef(mean1, mean2)[0, 1]
                if corr < 0:
                    X[:, :, k] = X[:, :, k] * -1
                    print(f'Flipped sign of label {k} in session {i} because correlation was negative')
            sessions_data[i] = (X, y)
    

    Xsesh, ysesh  = [], []

    for session in sessions_data:
        X = session[0]
        y = session[1]
        y = np.array([1 if i in animate_triggers else 0 for i in y])

        X, y, remove_ind = balance_class_weights(X, y)

        X = X.transpose(2,0,1)

        Xsesh.append(X)
        ysesh.append(y)

   
    Xbin, ybin, sesh_inds = [], [], []

    for n in range(7):
        x, y, sesh_inds_temp = create_blocks(sessions_data, n_bins=7, n=n, animate_triggers = animate_triggers)
        Xbin.append(x)
        ybin.append(y)
        sesh_inds.append(sesh_inds_temp)

    np.savez(f'data/xbin.npz', Xbin[0],Xbin[1],Xbin[2],Xbin[3],Xbin[4],Xbin[5],Xbin[6], allow_pickle = True)
    np.save(f'data/ybin.npy', np.array(ybin, dtype=object))
    np.save(f'data/ybin_seshinds.npy', np.array(sesh_inds, dtype=object))

if __name__ == '__main__':
    main()