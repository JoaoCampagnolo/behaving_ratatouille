from enum import Enum


class Behav(Enum):
    HAPINESS = 0
    SADNESS = 1
    FEAR = 2
    DISGUST = 3
    ANGER = 4
    SURPRISE = 5
    FOCUSED = 6
    REST = 7
    NONE = 8
    BOUNDARY = 9
    
# keep adding stereotyped labels for supervised training
    
# List of labels and frame intervals. Keep adding manual annotations
label_gt_list = [([0, 140], Behav.HAPINESS, 'data_acq_jhc/CAMPAGNOLO/Experiment_datetime/EEG_recording_2021-07-06-13.21.58'),
                 ((140, 460), Behav.SURPRISE, 'data_acq_jhc/CAMPAGNOLO/Experiment_datetime/EEG_recording_2021-07-06-13.21.58'),
                 ((600, 750), Behav.FOCUSED, 'data_acq_jhc/CAMPAGNOLO/Experiment_datetime/EEG_recording_2021-07-06-13.21.58'),
                 ((750, 900), Behav.HAPINESS, 'data_acq_jhc/CAMPAGNOLO/Experiment_datetime/EEG_recording_2021-07-06-13.21.58'),

                 ([0, 140], Behav.FEAR, 'data_acq_jhc/CAMPAGNOLO/Experiment_datetime/EEG_recording_2021-07-09-17.56.03'),
                 ((140, 500), Behav.SURPRISE, 'data_acq_jhc/CAMPAGNOLO/Experiment_datetime/EEG_recording_2021-07-09-17.56.03'),
                 ((630, 800), Behav.ANGER, 'data_acq_jhc/CAMPAGNOLO/Experiment_datetime/EEG_recording_2021-07-09-17.56.03'),
                 ((790, 900), Behav.HAPINESS, 'data_acq_jhc/CAMPAGNOLO/Experiment_datetime/EEG_recording_2021-07-09-17.56.03'),

                 ]

subjects_dir = ['/Users/joaohenrique/Documents/Clusterolo/NEVARO_behavior/muse_recordings/CAMPAGNOLO',
               '/Users/joaohenrique/Documents/Clusterolo/NEVARO_behavior/muse_recordings/RITA']


train_dir_single = ['/Users/joaohenrique/Documents/Clusterolo/NEVARO_behavior/muse_recordings/CAMPAGNOLO/2021_07_06/EEG_recording_2021-07-06-13.21.58.csv']

train_dir = []

val_dir = []

test_dir = []

train_colors = ['red', 'green', 'blue', 'orange', 'purple', 'palegreen', 'sienna', 'magenta']
val_colors = ['yellow', 'cyan', 'crimson', 'darkgrey', 'salmon', 'olive' , 'darkcyan']

            