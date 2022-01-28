from enum import Enum


class Tracked(Enum):
    SNOUT = 0
    RIGHT_EAR = 1
    LEFT_EAR = 2
    MIDSPINE = 3
    TAIL = 4


tracked_points = [Tracked.SNOUT, 
                  Tracked.RIGHT_EAR, Tracked.LEFT_EAR, 
                  Tracked.MIDSPINE,
                  Tracked.TAIL]

point_names = ['SNOUT', 'RIGHT_EAR', 'LEFT_EAR', 'MIDSPINE', 'TAIL']

is_x = [True, False, False,
         True, False, False, True, False, False,
         True, False, False,
         True, False, False]

is_y = [False, True, False,
         False, True, False, False, True, False,
         False, True, False,
         False, True, False]

is_score = [False, False, True,
            False, False, True, False, False, True,
            False, False, True,
            False, False, True]

limb_id = [0, 
           1, 1,
           2,
           3]

#__side_left = [True, True, False, False] 

#__side_right = [False, False, True, True]

#__side_frontal = [False, True, True, False]

#__side_parietal = [True, False, False, True]

#def is_right_side(headband_id):
#    return tracked_points[headband_id] == [Tracked.TP9 or Tracked.AF7]

#def is_left_side(headband_id):
#    return tracked_points[headband_id] == [Tracked.AF8 or Tracked.TP10]

#def is_frontal_side(headband_id):
#    return tracked_points[headband_id] == [Tracked.AF8 or Tracked.AF7]

#def is_parietal_side(headband_id):
#    return tracked_points[headband_id] == [Tracked.TP9 or Tracked.TP10]

#def point_names():
#    return ['SNOUT', 'RIGHT_EAR', 'LEFT_EAR', 'MIDSPINE', 'TAIL']

colors = [(255, 0, 0),
          (0, 0, 255),
          (0, 255, 0),
          (150, 200, 200),
          (50, 150, 150)]

num_points = len(tracked_points)

