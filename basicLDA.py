'''
The following script will be used to run basic linear discriminant analysis on EEG data for motor imagery classification
Author: George Hanna
Date : 18/11/2017
'''
import mne
import numpy as np
import matplotlib.pyplot as plt

#Data path
datapath = '/Users/georgehanna/Projects/MotorImagery/Data/A01T.gdf'
data_obj = mne.io.read_raw_edf(datapath, preload=True, stim_channel=None)

#Concatenate events info as a STIM channel in order to use MNE functions later
event_type = data_obj._raw_extras[0]['events'][2]
event_duration = data_obj._raw_extras[0]['events'][4]
event_latency = data_obj._raw_extras[0]['events'][1]
stim_data = np.zeros((1, len(data_obj.times)))

#TO-Do I just store onset times or do I incorporate duration? For now just doing onset
for i in range(len(event_type)):
    stim_data[0,event_latency[i]] = event_type[i]


info = mne.create_info(['STI'],data_obj.info['sfreq'],['stim'])
stim_raw = mne.io.RawArray(stim_data, info)
data_obj.add_channels([stim_raw],force_update_info =True)

print(data_obj)

#Visualize events (for debugging)
events = mne.find_events(data_obj,stim_channel='STI',verbose=True)
event_id = {"left": 769,"right":770, "foot":771,"tongue": 772, "EyeOpen": 276, "EyeClosed": 277, "StartTrial": 768, "Unknown"
: 783, "EyeMove": 1072, "Rejected":1023, "StartRun":32766 }

mne.viz.plot_events(events,data_obj.info['sfreq'],data_obj.first_samp, event_id =event_id, show=True)
print(events)