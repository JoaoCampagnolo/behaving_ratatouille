# File:             trial_selection.py
# Date:             October 2021
# Description:      Allows the user to assess the available patient files (EEG recordings or other). Following this, the user
#                   is free to decide which files should proceed along the pipeline
# Authors:          Joao Campagnolo
# Python version:   Python 3.7+

# Import packages:
import os
import pandas as pd
#import glob

def patient_files(path):
    ignored = '.ipynb_checkpoints' #ignoring annoying notebook checkpoints
    if os.path.isdir(path):
        d = {}
        for name in [name for name in os.listdir(path) if name not in ignored]:
            d[name] = patient_files(os.path.join(path, name))
    else:
        d = os.path.getsize(path)
    return d

# root_dir needs a trailing slash (i.e. /root/dir/)
#for filename in glob.iglob(root_dir + '**/*.csv', recursive=True):
#     print(filename)