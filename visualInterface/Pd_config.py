import os
from datetime import datetime

# 4 = Fixation
# 8 = ND 
# 32 = D right
# 44 = D left
# 64 = response


MODE = 'test' # train, decode

# Stimuli configuration
viewing_distance_cm = 75.0
eccentricity_deg = 4        # d from fixation to shape
shape_definitions = [
    {"type": "diamond", "size_deg": (2, 2)},   # width x height
    {"type": "circle", "size_deg": 2},            # diameter
    {"type": "hexagon", "size_deg": (2, 2)},    # width x height
    {"type": "square", "size_deg": (1.7, 1.7)}     # width x height
    
]
set_size = len(shape_definitions)
font = 'Calibri'
font_size = 40
fixation_size = 12
line_width = 5
dot_radius = 7
icon_size = (150, 150)

# Display configuration
#   mac
# screen_width_cm = 28.66
# screen_height_cm = 17.92
#thinkpad
screen_width_cm = 30.5
screen_height_cm = 18.0 


# Experiment parameters
n_trials = 60
n_d_trials = n_trials/2
break_trial = 31  # trial index at which to take a break
delay_duration_ms = [350, 750, 50] # min, max, step
fixation_duration_ms = 1000
stimulus_duration_ms = 2000
break_duration_ms = 10000
feedback_duration_ms = 500

#Classification targets
TPRr = 75 #%
TPRl = 75 #%
TNR = 75 #%

