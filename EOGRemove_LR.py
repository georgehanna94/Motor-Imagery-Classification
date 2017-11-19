'''
The following function will be used to implement EOG artifact removal from EEG channels using Linear Regression
according to the technique described in "A Fully Automated correction method of EOG artifacts in EEG recordings by Schlogl et al"

Author: George Hanna
Date : Nov. 19th 2017
'''

import numpy as np

def EOGRemove(Noisy_EEG, EOG):

    #Check that matrix lengths are correct
    assert Noisy_EEG.shape[0] == EOG.shape[0]

    #Compute b (weights of artifacts per channel) - size is 3 (EOG channels) x 22 (EEG Channels)
    Cnn = np.dot(EOG.T,EOG) #auto-covariance matrix of the EOG channels
    Cny = np.dot(EOG.T,Noisy_EEG)
    b = np.dot(np.linalg.inv(Cnn),Cny)

    #Subtract artifact from noisy signal
    cleanEEG = Noisy_EEG - np.dot(EOG,b)

    return cleanEEG