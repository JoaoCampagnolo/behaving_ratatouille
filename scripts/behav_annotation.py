from enum import Enum


class Behav(Enum):
    WALK_FORW = 0
    WALK_BACKW = 1
    STAND_TWO_LEGS = 2
    REST = 3
    GROOM = 4
    LEFT_TURN = 5
    RIGHT_TURN = 6
    NONE = 7
    BOUNDARY = 8
    
# keep adding stereotyped labels for supervised training
    
# List of labels and frame intervals. Keep adding manual annotations
label_gt_list = [((0, 140), Behav.WALK_FORW, "KI_article\\scripts\\POSE-JoaoCampagnolo-2021-12-13"),
                 ((140, 460), Behav.WALK_BACKW, "KI_article\\scripts\\POSE-JoaoCampagnolo-2021-12-13"),
                 ((600, 750), Behav.STAND_TWO_LEGS, "KI_article\\scripts\\POSE-JoaoCampagnolo-2021-12-13"),
                 ((750, 900), Behav.REST, "KI_article\\scripts\\POSE-JoaoCampagnolo-2021-12-13"),

                 ((0, 140), Behav.GROOM, "KI_article\\scripts\\POSE-JoaoCampagnolo-2021-12-09"),
                 ((140, 500), Behav.LEFT_TURN, "KI_article\\scripts\\POSE-JoaoCampagnolo-2021-12-09"),
                 ((630, 800), Behav.RIGHT_TURN, "KI_article\\scripts\\POSE-JoaoCampagnolo-2021-12-09"),
                 ((790, 900), Behav.BOUNDARY, "KI_article\\scripts\\POSE-JoaoCampagnolo-2021-12-09"),

                 ]

all_dir = ["C:\\Users\\jhflc\\OneDrive\\Documentos\\Projects\\KI_article\\scripts\\POSE-JoaoCampagnolo-2021-12-13",
           "C:\\Users\\jhflc\\OneDrive\\Documentos\\Projects\\KI_article\\scripts\\POSE-JoaoCampagnolo-2021-12-09"] #keep adding


train_dir_single = ["C:\\Users\\jhflc\\OneDrive\\Documentos\\Projects\\KI_article\\scripts\\POSE-JoaoCampagnolo-2021-12-13"]

train_dir = ["C:\\Users\\jhflc\\OneDrive\\Documentos\\Projects\\KI_article\\scripts\\POSE-JoaoCampagnolo-2021-12-13"]

train_names = ['375529_2021-07-25_7']

val_dir = []

test_dir = []

mouse_tags = []

train_colors = ['red', 'green', 'blue', 'orange', 'purple', 'palegreen', 'sienna', 'magenta']
val_colors = ['yellow', 'cyan', 'crimson', 'darkgrey', 'salmon', 'olive' , 'darkcyan']

            