import pygame
import sys
import random
import math
from python_client import Trigger

# Trigeers
# fixation 6
# dot top 100
# dot right 101
# dot bottom 102
# dot left 103

def degrees_to_pixels(degrees, viewing_distance_cm, pixels_per_cm):
    radians = math.radians(degrees)
    size_cm = 2 * viewing_distance_cm * math.tan(radians / 2)
    size_px = size_cm * pixels_per_cm
    return size_px

# Initialize pygame
pygame.init()
pygame.font.init()

def add_trigger(code):
    timestamp = pygame.time.get_ticks()
    trigger.append((code, timestamp, trial_index))
    if code != None:
        HWTrigger.signal(code)

HWTrigger = Trigger('USB2LPT')
HWTrigger.init(50)
trigger=[]

# Setup display (full-screen)
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width, screen_height = screen.get_size()
x_center = screen_width // 2
y_center = screen_height // 2

# Physical parameters (adjust as needed)
screen_width_cm = 30.5    # cm
screen_height_cm = 18.0   # cm
viewing_distance_cm = 60.0
# Compute average pixels per cm from screen dimensions
pixels_per_cm = ((screen_width / screen_width_cm) + (screen_height / screen_height_cm)) / 2

# Define eccentricity and compute pixel distance from center
eccentricity_deg = 5      # degrees of visual angle
d_from_center = degrees_to_pixels(eccentricity_deg, viewing_distance_cm, pixels_per_cm)

# Define four dot positions (top, right, bottom, left relative to center)
positions = [
    (x_center, y_center - d_from_center),  # top
    (x_center + d_from_center, y_center),  # right
    (x_center, y_center + d_from_center),  # bottom
    (x_center - d_from_center, y_center)   # left
]

# Create trial list: 10 trials per location (0=top, 1=right, 2=bottom, 3=left)
trials = []
for pos_index in range(4):
    trials.extend([pos_index] * 10)
random.shuffle(trials)  # Randomize trial order

# Set up font for instructions
font = pygame.font.SysFont('Calibri', 40)

# Wait for key press to start the calibration
screen.fill((0, 0, 0))
start_text = font.render('Press any key to start calibration', True, (255, 255, 255))
text_rect = start_text.get_rect(center=(x_center, y_center))
screen.blit(start_text, text_rect)
pygame.display.flip()

waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            waiting = False
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

# Main calibration loop (40 trials total)
trial_index = 0
total_trials = len(trials)

while trial_index < total_trials:
    # Allow quitting with ESC anytime during trials
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()
    
    # 1. Fixation cross for 500 msec
    screen.fill((0, 0, 0))
    pygame.display.flip()
    pygame.time.delay(500)
    fixation_size = 20
    line_width = 3
    # Horizontal line
    pygame.draw.line(screen, (255, 255, 255),
                     (x_center - fixation_size, y_center),
                     (x_center + fixation_size, y_center),
                     line_width)
    # Vertical line
    pygame.draw.line(screen, (255, 255, 255),
                     (x_center, y_center - fixation_size),
                     (x_center, y_center + fixation_size),
                     line_width)
    pygame.display.flip()
    add_trigger(6)
    pygame.time.delay(1000)
    
    # 2. Calibration dot appears for 1000 msec
    pos_index = trials[trial_index]
    dot_x, dot_y = positions[pos_index]
    screen.fill((0, 0, 0))
    dot_radius = 10  # You can adjust the dot size here
    pygame.draw.circle(screen, (255, 255, 255), (int(dot_x), int(dot_y)), dot_radius)
    pygame.display.flip()

    if pos_index == 0:
        add_trigger(100)
    if pos_index == 1:
        add_trigger(101)
    if pos_index == 2:
        add_trigger(102)
    if pos_index == 3:
        add_trigger(103) 
    pygame.time.delay(1000)
    
    trial_index += 1

basename = sys.argv[1]
trigger_file = f"{basename}.triggers.txt"
with open(trigger_file, "w") as trigger_txt:
    for code, timestamp, trial in trigger:
        trigger_txt.write(f"{trial+1} {code} {timestamp}\n")

# End-of-task message
screen.fill((0, 0, 0))
end_text = font.render('Calibration complete. Press any key to exit.', True, (255, 255, 255))
text_rect = end_text.get_rect(center=(x_center, y_center))
screen.blit(end_text, text_rect)
pygame.display.flip()

waiting = True
while waiting:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            waiting = False
        if event.type == pygame.QUIT:
            waiting = False

pygame.quit()
sys.exit()
