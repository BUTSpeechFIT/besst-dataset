import numpy as np
participants = np.loadtxt("dataset/metadata/participants.csv", dtype=object, delimiter=";", skiprows=1)
multilingual_data_available_mask = np.all(participants[:, 6:].astype(int), axis=1)
filtered_participants = participants[multilingual_data_available_mask, :6]


speech_segments = np.loadtxt("dataset/metadata/speech-segments.csv", dtype=object, delimiter=";", skiprows=1)

speech_segments_for_participants = [sps for sps in speech_segments if sps[0] in filtered_participants[:,0]]
speech_segments_for_participants_cl =  np.array([sps for sps in speech_segments_for_participants if sps[6] == 'Rebus'])


np.unique(speech_segments_for_participants_cl[:,5], return_counts=True)




