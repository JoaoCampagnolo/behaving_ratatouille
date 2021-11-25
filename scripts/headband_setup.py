from enum import Enum


class Tracked(Enum):
    TP9 = 0
    AF7 = 1
    AF8 = 2
    TP10 = 3


tracked_points = [Tracked.TP9, Tracked.AF7, Tracked.AF8, Tracked.TP10]

headband_id = [0, 1, 2, 3]

__side_left = [True, True, False, False] 

__side_right = [False, False, True, True]

__side_frontal = [False, True, True, False]

__side_parietal = [True, False, False, True]

def is_right_side(headband_id):
    return tracked_points[headband_id] == [Tracked.TP9 or Tracked.AF7]

def is_left_side(headband_id):
    return tracked_points[headband_id] == [Tracked.AF8 or Tracked.TP10]

def is_frontal_side(headband_id):
    return tracked_points[headband_id] == [Tracked.AF8 or Tracked.AF7]

def is_parietal_side(headband_id):
    return tracked_points[headband_id] == [Tracked.TP9 or Tracked.TP10]

def electrode_names():
    return ['TP9', 'AF7', 'AF8', 'TP10']

colors = [(255, 0, 0),
          (0, 0, 255),
          (0, 255, 0),
          (150, 200, 200)]

num_electrodes = len(tracked_points)

