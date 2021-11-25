# File:             preprocess.py
# Date:             Winter 2021
# Description:      Some utils that are usefull to train a DLC model.
# Authors:          Joao Campagnolo
# Python version:   Python 3.7+

# Import packages
import os

def find_videos(root_dir, extension='.avi'):
    '''
    Root dir should be the path that is common to all data files, that follow a common structure: Data<Recording<files>>.
    Targets files with a given extension, eg. .avi, .pkl, etc.
    '''
    animals = {}
    animals_list = [] 
    for exp in [f for f in os.listdir(root_dir) if not os.path.isfile(os.path.join(root_dir, f))]: #gets all recording folder names
        subj_path = root_dir+'/'+exp
        for folder in [f for f in os.listdir(subj_path) if not os.path.isfile(os.path.join(subj_path, f))]: #checks all animal folders
            for fname in os.listdir(subj_path+'/'+folder):
                if fname.endswith(extension):
                    animals_list.append(fname)
                    animal_id = fname.split('_')[1]
                    if animal_id not in list(animals.keys()):
                        animals[animal_id] = fname
                    else:
                        animals[fname.split('_')[1]].append(fname)
    return animals

def get_paths(file_names, root_dir):
    '''
    Starts with a list of file names with a common root directory and outputs their full paths.
    '''
    paths_list = list()
    for file in file_names:
        for root, dirs, files in os.walk(root_dir):
            for name in files:
                if name == file:
                    paths_list.append(os.path.abspath(os.path.join(root, name)))
    return paths_list