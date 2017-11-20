'''
The following script will be used to run basic linear discriminant analysis on EEG data for motor imagery classification
Author: George Hanna
Date : 18/11/2017
'''
import mne
from mne.decoding import CSP
import numpy as np
import EOGRemove_LR
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

'''Load Data'''
'''-------------------------------------------'''
#Data path
datapath = '/Users/georgehanna/Projects/MotorImagery/Data/A01T.gdf'
data_obj = mne.io.read_raw_edf(datapath, preload=True, stim_channel=None)
#Visualize Data channels
#data_obj.plot()

#Concatenate events info as a STIM channel in order to use MNE functions later
event_type = data_obj._raw_extras[0]['events'][2]
event_duration = data_obj._raw_extras[0]['events'][4]
event_latency = data_obj._raw_extras[0]['events'][1]
stim_data = np.zeros((1, len(data_obj.times)))

'''Here the event code is stored at the latency time specified by event_latency
Note that if a trial is rejected, (code 1023) will appear at the same latency as as the original event code (768)
'''
for i in range(len(event_type)):
    #if the trial is rejected, don't store the event code
    if (i!=0 and event_type[i-1] != 1023):
        stim_data[0,event_latency[i]] = event_type[i]


info = mne.create_info(['STI'],data_obj.info['sfreq'],['stim'])
stim_raw = mne.io.RawArray(stim_data, info)
data_obj.add_channels([stim_raw],force_update_info =True)

#print(data_obj)

'''EOG Removal'''
'''--------------------------------------'''
#Extract EEG only channels
Noisy_EEG = np.transpose(data_obj.get_data()[0:22,:])
#Extract EOG only channels
EOG = np.transpose(data_obj.get_data()[22:25,:])
#Return cleaned up EEG and store it back into data object
cleanEEG = EOGRemove_LR.EOGRemove(Noisy_EEG,EOG)
data_obj._data[0:22] = cleanEEG.T

#data_obj.plot()

'''Epoch Dataset'''
'''--------------------------------------'''
#Visualize events (for debugging)
events = mne.find_events(data_obj,stim_channel='STI',verbose=True)
event_id = {"left": 769,"right":770, "foot":771,"tongue": 772 }
#mne.viz.plot_events(events,data_obj.info['sfreq'],data_obj.first_samp, event_id =event_id, show=True)

#Define time constraints on epochs (based on data collection paradigm defined in dataset PDF)
tmin, tmax = 0., 4.
#Pick EEG channels from dataset
picks = np.arange(0,22)

# Extract epochs and plot for a single Motor imagery event across all channels
epochs = mne.Epochs(data_obj, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)
epochs_data = epochs.get_data()
#plt.plot(1e3*epochs.times, 1e6*epochs_data[0,:,:].T)
#plt.xlabel('Time(ms)')
#plt.ylabel('Epoch Data for Tongue MI (microV)')

'''Do CSP on Epochs to get features and LDA for Classification: code by Martin Billinger '''
#TODO FIGURE OUT WHY CLASSIFICATION ACCURACY SUCKS
'''----------------------------------------'''
# Define a monte-carlo cross-validation generator (reduce variance):
scores = []
epochs_train = epochs.copy().crop(tmin=1., tmax=3.5)
epochs_data_train = epochs_train.get_data()
labels1 = epochs.events[:, -1] - 2
labels = labels1 - 769
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
cv_split = cv.split(epochs_data_train)

# Assemble a classifier
lda = LinearDiscriminantAnalysis()
csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

# Use scikit-learn Pipeline with cross_val_score function
clf = Pipeline([('CSP', csp), ('LDA', lda)])
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

# Printing the results
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))

print("done")
