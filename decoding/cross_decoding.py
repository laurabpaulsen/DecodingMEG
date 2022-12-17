"""
Use this script to run cross decoding either in sensor space or source space.

usage: cross_decoding.py [-h] [--sens SENS]
"""

import mne 
import numpy as np
import decoder_cross as decoders
import json
import multiprocessing as mp
from datetime import datetime
import time 
from decoding import prep_data
import argparse as ap

classification = True
ncv = 5
ncores = 10 #mp.cpu_count()
alpha = 'auto'
model_type = 'LDA' # can be either LDA, SVM or RidgeClassifier
now = datetime.now()
output_path = f'./accuracies/cross_decoding_ncv_{ncv}.npy'

def get_accuracy(input:tuple, classification=classification, ncv=ncv):
    decoder = decoders.Decoder(classification=classification, ncv = ncv, alpha = alpha, scale = True, model_type = model_type, get_tgm=True)
    (session_train, session_test, idx) = input

    if session_test == session_train: # avoiding double dipping within session, by using within session decoder
        X = Xsesh[session_train]
        y = ysesh[session_train]
        accuracy = decoder.run_decoding(X, y)

    else:
        X_train = Xsesh[session_train]
        X_test = Xsesh[session_test]

        y_train = ysesh[session_train]
        y_test = ysesh[session_test]

        accuracy = decoder.run_decoding_across_sessions(X_train, y_train, X_test, y_test)
    print(f'Index {idx} done')

    return session_train, session_test, accuracy



if __name__ == '__main__':
    parser = ap.ArgumentParser()
    # add sens as an argument, true or false
    parser.add_argument('--sens', type=bool, help='True if you want to use sensor space, False if you want to use source space')
    args = parser.parse_args()
    sens = args.sens

    if not sens:
        output_path = f'./accuracies/cross_decoding_ncv_{ncv}.npy'
    else:
        output_path = f'./accuracies/cross_decoding_sens_ncv_{ncv}.npy'


    st = time.time()
    Xbin, ybin, Xsesh, ysesh = prep_data(sens = sens)
    del Xbin, ybin

    Xsesh = [np.concatenate(i, axis = 2) for i in Xsesh]
    Xsesh = [np.transpose(i.squeeze(), (0,1,2)) for i in Xsesh]
    ysesh = [np.concatenate(i, axis = 0) for i in ysesh]

    for i in range(len(Xsesh)):
        print(Xsesh[i].shape, ysesh[i].shape)

    decoding_inputs = [(train_sesh, test_sesh, idx) for idx, train_sesh in enumerate(range(len(Xsesh))) for test_sesh in range(len(Xsesh))]
    
    accuracies = np.zeros((len(Xsesh), len(Xsesh), 250, 250), dtype=float)
    with mp.Pool(ncores) as p:
        for train_session, test_session, accuracy in p.map(get_accuracy, decoding_inputs):
            accuracies[train_session, test_session, :, :] = accuracy
    
    p.close()
    p.join()


    
    et = time.time()
    print(f'Time taken: {et-st}')

    np.save(output_path, accuracies)