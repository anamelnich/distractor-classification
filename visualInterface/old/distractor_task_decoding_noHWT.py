
import pygame
import math
import random
import os
from datetime import datetime
import sys
from pygame.locals import *
###### These are the imports to make the hardware triggers work ######
# from python_client import Trigger
from cnbiloop import BCI_tid ##############new
import threading
import subprocess



########################functions########################
def degrees_to_pixels(degrees, viewing_distance_cm, pixels_per_cm):
    radians = math.radians(degrees)
    size_cm = 2 * viewing_distance_cm * math.tan(radians / 2)
    size_px = size_cm * pixels_per_cm
    return size_px
def add_trigger(code):
    timestamp = pygame.time.get_ticks()
    trigger.append((code, timestamp, trial_index))
    # if code != None:
    #     HWTrigger.signal(code)
def send_tid(value):
    bci.id_msg_bus.SetEvent(value)
    bci.iDsock_bus.sendall(str.encode(bci.id_serializer_bus.Serialize()))
def receiveTiD():
    global bci
    data = None
    try:
        data = bci.iDsock_bus.recv(512).decode("utf-8")  # Replaced curly quotes with straight quotes
        bci.idStreamer_bus.Append(data)
    except BlockingIOError as e:
        if e.errno != 11:  # Ignore Resource temporarily unavailable errors
            print("BlockingIOError in receiveTiD:", e)
    except Exception as e:
        print("Error in receiveTiD:", e)
    
    # Deserialize ID message
    if data:
        if bci.idStreamer_bus.Has("<tobiid", "/>"):
            msg = bci.idStreamer_bus.Extract("<tobiid", "/>")
            bci.id_serializer_bus.Deserialize(msg)
            bci.idStreamer_bus.Clear()
            tmpmsg = int(round(float(bci.id_msg_bus.GetEvent())))
            print(f"Received TID Message: {tmpmsg}")
            return tmpmsg

        elif bci.idStreamer_bus.Has("<tcstatus", "/>"):
            MsgNum = bci.idStreamer_bus.Count("<tcstatus")
            for i in range(1, MsgNum-1):
                # Extract and discard unwanted <tcstatus> messages
                msg_useless = bci.idStreamer_bus.Extract("<tcstatus", "/>")
# Function to continuously receive triggers in a separate thread
def trigger_listener(running_flag):
    
    while running_flag[0]:  # controls execution of thread, logical list
        msg = receiveTiD()  # Call the older function
        if msg:
            if msg == 2:
                print('Pd Detected')
                # Update global flags or perform actions as needed
                # Example:
                global correct_detected
                correct_detected = True
            elif msg == 1:
                print('No Pd Detected')
                # Update global flags or perform actions as needed
                global error_detected, feedback_mode
                error_detected = True
                feedback_mode = True
        pygame.time.wait(20)  # Wait for 20 milliseconds
def circle(x, y, radius):
    pygame.draw.circle(screen, (0, 128, 0), (int(x), int(y)), int(radius))

def square(x, y, width, height, color):
    pygame.draw.rect(screen, color, (int(x - width / 2), int(y - height / 2), int(width), int(height)))

def diamond(x, y, width, height, color):
    pygame.draw.polygon(screen, color, [
        (int(x), int(y - height / 2)),  # Top
        (int(x + width / 2), int(y)),   # Right
        (int(x), int(y + height / 2)),  # Bottom
        (int(x - width / 2), int(y))    # Left
    ])
def hexagon(x, y, width, height, color):
    radius = width / 2
    points = [(x + radius * math.cos(math.radians(60 * i)), y + radius * math.sin(math.radians(60 * i))) for i in range(6)]
    pygame.draw.polygon(screen, color, [(int(px), int(py)) for px, py in points])

def draw_dot(x, y, shape_width, side):
    dot_radius = 5
    if side == 1:  # Right
        dot_x = int(x + shape_width * 0.25)
        dot_y = int(y)
    elif side == 0:  # Left
        dot_x = int(x - shape_width * 0.25)
        dot_y = int(y)

    pygame.draw.circle(screen, (0, 0, 0), (dot_x, dot_y), dot_radius)

# Trigger list
# 50 = start trigger

# Fixation
# 60 = Fixation
# 70 = feedback 

#Onset triggers
# 10 = ND UP (11 correct, 12 incorrect, 13 timeout)
# 20 = D UP (21 correct, 22 incorrect, 23 timeout)


pygame.init()
pygame.font.init()




########################innit vars########################
timestamp_date = datetime.now().strftime("%Y%m%d")
timestamp = datetime.now().strftime("%Y%m%d%H%M")
# output_folder = f"saved_files/distractor/e{subject_number}_{timestamp_date}/e{subject_number}_{timestamp}{subject_run}_training"
# os.makedirs(output_folder, exist_ok=True)
# gdf_file = os.path.join(output_folder, f"e{subject_number}_{timestamp}{subject_run}_training.gdf")
# log_file = os.path.join(output_folder, f"e{subject_number}_{timestamp}{subject_run}_training.log")
text_to_save = ""
text_to_analyze = ""
# # Open GDF and log files
# args = ["cl_rpc", "openxdf", gdf_file, log_file, "\"\""]
# subprocess.run(args) 
# ##### This part initialize triggers list of hardware triggers to send to the amplifier #####
# HWTrigger = Trigger('USB2LPT')
# HWTrigger.init(50)
bci = BCI_tid.BciInterface() 
trial_index = 0
trigger=[]
responses = []
n_trials = 60
n_d_trials = n_trials/2


########################define shape sizes and location based on eccentricity + other graphic elements########################
# screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
# screen_width_px, screen_height_px = screen.get_size()
# Replace the FULLSCREEN mode with RESIZABLE mode
screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)
pygame.display.set_caption("D Task")
screen_width_px, screen_height_px = 1920, 1080

thumb_up = pygame.image.load("./img/thumb_up.png").convert_alpha()
thumb_down = pygame.image.load("./img/thumb_down.png").convert_alpha()
screen_width_cm = 30.5  # in cm
screen_height_cm = 18.0 
viewing_distance_cm = 60.0
eccentricity_deg = 5

pixels_per_cm_x = screen_width_px / screen_width_cm
pixels_per_cm_y = screen_height_px / screen_height_cm
pixels_per_cm = (pixels_per_cm_x + pixels_per_cm_y) / 2
x_center = screen_width_px / 2
y_center = screen_height_px / 2
d_from_center = degrees_to_pixels(eccentricity_deg, viewing_distance_cm, pixels_per_cm)

shape_definitions = [
    {"type": "diamond", "size_deg": (1.3, 1.3)},   # width x height
    {"type": "circle", "size_deg": 1.3},            # diameter
    {"type": "hexagon", "size_deg": (1.3, 1.3)},    # width x height
    {"type": "square", "size_deg": (1.2, 1.2)},     # width x height
]
# Convert shape sizes from degrees to pixels
for shape in shape_definitions:
    if shape["type"] == "circle":
        # For circle, size_deg represents diameter
        shape["diameter_px"] = degrees_to_pixels(shape["size_deg"], viewing_distance_cm, pixels_per_cm)
        shape["radius_px"] = shape["diameter_px"] / 2
    else:
        # For other shapes, size_deg represents width and height
        width_deg, height_deg = shape["size_deg"]
        shape["width_px"] = degrees_to_pixels(width_deg, viewing_distance_cm, pixels_per_cm)
        shape["height_px"] = degrees_to_pixels(height_deg, viewing_distance_cm, pixels_per_cm)

# Calculate shape coordinates
set_size = len(shape_definitions)
shape_coord = []
start_angle_offset = math.pi / 2  # 90 degrees to start at the top

for i in range(set_size):
    angle = i * (2 * math.pi / set_size) - start_angle_offset
    x = x_center + d_from_center * math.cos(angle)
    y = y_center + d_from_center * math.sin(angle)
    shape_coord.append((x, y))
if set_size == 4:
    mid_pos = [1,3]
    lat_pos = [2,4]
elif set_size == 6:
    mid_pos = [1,4]
    lat_pos = [2,3,5,6]
elif set_size == 10:
    mid_pos = [1,6]
    lat_pos = [2,3,4,5,7,8,9,10]

font = pygame.font.SysFont('Calibri', 40)
fixation_size = 12
line_width = 5


########################randomize trials and trial parameters########################

shape_positions = []

trial_type = [0] * 30 + [1] * 30  # 0 = no distractor, 1 = distractor
random.shuffle(trial_type)
while any(trial_type[i] == trial_type[i+1] == trial_type[i+2] == trial_type[i+3] for i in range(len(trial_type) - 3)):
    random.shuffle(trial_type)  # Reshuffle if four or more identical values in a row

#generate distractor positions
d_pos = [0] * len(trial_type)
t_pos = [0]*len(trial_type)

all_positions = lat_pos * math.ceil((n_d_trials)/len(lat_pos)) #2 * (30/2) generates 30
random.shuffle(all_positions)

for i in range(len(trial_type)):
    if trial_type[i] == 1:  # Only assign for distractor trials
        d_pos[i] = all_positions.pop(0)  # Assign and remove from available


#generate target positions
all_latd_t_pos = mid_pos * int(n_d_trials/2) #30
random.shuffle(all_latd_t_pos)

for i in range(n_trials):
    if trial_type[i] == 1:
        t_pos[i]=all_latd_t_pos.pop(0)
    if trial_type[i] == 0:
        available_positions = [pos for pos in range(1, set_size + 1)]
        chosen_pos = random.choice(available_positions)
        t_pos[i] = chosen_pos

# Generate shape_positions
for i in range(n_trials):
    # Randomize shape positions
    available_positions = [pos for pos in range(1, set_size+1) if pos != t_pos[i]]
    random.shuffle(available_positions)

    shape_names = ['Diamond', 'Hexagon', 'Square','Diamond', 'Hexagon', 'Square','Diamond', 'Hexagon','Square']
    shape_positions_trial = {}
    shape_positions_trial[t_pos[i]] = ('Circle', random.choice([0, 1]))  # Circle with random dot side

    shape_names = shape_names[:set_size-1]
    for j, shape in enumerate(shape_names):
        dot_side = random.choice([0, 1])  # 0=left, 1=right
        shape_positions_trial[available_positions[j]] = (shape, dot_side)

    shape_positions.append(shape_positions_trial)


big_font = pygame.font.Font(None, 72) 
small_font = pygame.font.Font(None, 36)
correct_detected = False
error_detected = False
feedback_mode = False
cmTP = 0
cmTN = 0
cmFP = 0
cmFN = 0
icon = None
score=0
# Initialize running flag for the listener thread
listener_running = [True]

# Start the trigger listener thread
#threading allows to run functions in parallel, so here allows to continuously check for triggers WHILE the main loop is running
listener_thread = threading.Thread(target=trigger_listener, args=(listener_running,)) 
listener_thread.start()

#main loop checks for events continuously 
run = True
task_start_time = pygame.time.get_ticks()


while run:
    
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            run = False
            send_tid(20)

    if trial_index < n_trials:
        if trial_index == 0:
            screen.fill((0, 0, 0))
            text = font.render(f'press any key to start', False, (225, 225, 225))
            textRect = text.get_rect()
            textRect.center = (x_center, y_center)
            screen.blit(text, textRect)
            pygame.display.update()
            waiting_for_key = True
            print("Waiting for key press to start...")
            while waiting_for_key:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        waiting_for_key = False
                        print("Key pressed. Starting trials...")
        #Draw blank screen
        trial_start = pygame.time.get_ticks()
        screen.fill((0, 0, 0))
        pygame.display.update()
        pygame.time.delay(500)

        # Draw fixation cross
        pygame.draw.line(screen, (225, 225, 225), (x_center - fixation_size, y_center), (x_center + fixation_size, y_center), line_width)
        pygame.draw.line(screen, (225, 225, 225), (x_center, y_center - fixation_size), (x_center, y_center + fixation_size), line_width)
        pygame.display.update()
        add_trigger(6)
        pygame.time.delay(1000)  # Fixation cross duration       

        # Track the start time of the trial
        array_start_time = pygame.time.get_ticks()
        trial_end = False  # Flag to end the trial
        if t_pos[trial_index] in lat_pos:
            t_side = 1
        elif t_pos[trial_index] in mid_pos:
            t_side = 0
        # if d_pos[trial_index]==10:
        #     t_side+=1
        #     d_pos[trial_index]=1
        if trial_type[trial_index]==1:
            start_trigger = int('1' + str(t_side)+str(d_pos[trial_index]))
        else:
            start_trigger = int('1' + str(t_side)+str(d_pos[trial_index]))
        add_trigger(start_trigger)
        wait = False
        while not trial_end:

            # Check if 2000ms have passed
            if pygame.time.get_ticks() - array_start_time >= 2000:
                trial_end = True
                response = 3
                add_trigger(13)
            elif wait:
                screen.fill((0, 0, 0))
                pygame.display.update()
            else:
            
                for i, (x, y) in enumerate(shape_coord):
                    shape_type, dot_side = shape_positions[trial_index][i + 1]
                    if trial_type[trial_index] == 1 and d_pos[trial_index] == i+1:
                        color = (255, 0, 0)
                    else:
                        color = (0, 128, 0)

                    if shape_type == 'Circle':
                        shape_radius = shape_definitions[1]["radius_px"]  # Circle's radius is at index 1
                        circle(x, y, shape_radius)  # Pass the radius in pixels
                        shape_width = shape_definitions[1]["diameter_px"]
                        dot_correct = dot_side

                    elif shape_type == 'Square':
                        shape_width = shape_definitions[3]["width_px"]    # Square's width in pixels
                        shape_height = shape_definitions[3]["height_px"]  # Square's height in pixels
                        square(x, y, shape_width, shape_height, color)

                    elif shape_type == 'Diamond':
                        shape_width = shape_definitions[0]["width_px"]    # Diamond's width in pixels
                        shape_height = shape_definitions[0]["height_px"]  # Diamond's height in pixels
                        diamond(x, y, shape_width, shape_height, color)

                    elif shape_type == 'Hexagon':
                        shape_width = shape_definitions[2]["width_px"]    # Hexagon's width in pixels
                        shape_height = shape_definitions[2]["height_px"]  # Hexagon's height in pixels
                        hexagon(x, y, shape_width, shape_height, color)
                    draw_dot(x, y, shape_width, dot_side)


                pygame.display.update()
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN :
                        if event.key == pygame.K_ESCAPE:
                            run = False
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        trial_end = False #True
                        wait = True
                        
                        if event.button == 1:
                            is_correct = (dot_correct == 0)
                        elif event.button == 3:
                            is_correct = (dot_correct == 1)

                        response = 1 if is_correct else 2
                        # if trial_type[trial_index] == 1:
                        #     trigger_code = 21 if is_correct else 22
                        # else:
                        trigger_code = 11 if is_correct else 12
                        
                        add_trigger(trigger_code)
                        screen.fill((0, 0, 0))
                        pygame.display.update()
        
        # After the trial ends, clear the screen
        screen.fill((0, 0, 0))
        if correct_detected:
            Pd_class = 1
            if trial_type[trial_index]==1:
                response_text = 'Score +2'
                response_color = (0,255,0)
                icon = thumb_up
                cmTP+=1
                score+=2
            elif trial_type[trial_index]==0:
                response_text = 'Score -1'
                response_color = (255,0,0)
                icon = thumb_down
                cmFP+=1
                score-=1
            correct_detected = False
        elif error_detected:
            Pd_class = 0
            if trial_type[trial_index]==1:
                response_text = 'Score -2'
                response_color = (255,0,0)
                icon = thumb_down
                cmFN+=1
                score-=2
            elif trial_type[trial_index]==0:
                response_text = 'Score +1'
                response_color = (0,255,0)
                icon = thumb_up
                cmTN+=1
                score+=1
            error_detected = False
        else:
            Pd_class = 3
            response_text = 'No Response Received'
            response_color = (255,255,255)
            icon = thumb_down
        
        
        # text = font.render(response_text, False, response_color)
        # textRect = text.get_rect()
        # textRect.center = (x_center, y_center)
        # screen.blit(text, textRect)
        if icon is not None:
            icon_rect = icon.get_rect()
            icon_rect.center = (x_center, y_center)
            screen.blit(icon, icon_rect)
        text = font.render(response_text, True, response_color)
        textRect = text.get_rect()
        textRect.midtop = (x_center, icon_rect.bottom + 10)
        screen.blit(text, textRect)
        pygame.display.update()
        pygame.time.delay(1000)
        icon=None
        responses.append(response)

        text_to_save += f"Trial {trial_index + 1} - Task: {trial_type[trial_index]} - Feedback: {response}\n"
        text_to_analyze += f"{trial_index + 1} {trial_type[trial_index]} {response} {t_pos[trial_index]} {d_pos[trial_index]} {dot_correct} {Pd_class}\n"
        

        trial_index += 1
    else:
        correct_responses = responses.count(1)/n_trials
        incorrect_response = responses.count(2)/n_trials
        timeout_response = responses.count(3)/n_trials
        screen.fill((0, 0, 0))
        score_text = big_font.render(f'Score: {score}', True, (255, 255, 255))
        score_rect = score_text.get_rect(center=(x_center, y_center))
        screen.blit(score_text, score_rect)
        text = font.render(f'Correct: {correct_responses}    Incorrect: {incorrect_response}     Timeout: {timeout_response}', True, (225, 225, 225))
        textRect = text.get_rect()
        textRect.centerx = x_center
        textRect.top = score_rect.bottom + 10
        screen.blit(text, textRect)
        pygame.display.update()
        waiting_for_key = True
        while waiting_for_key:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    waiting_for_key = False
        send_tid(20)
        run = False  # End the loop after all trials are completed

total = cmTP + cmFP + cmFN + cmTN

print("Confusion Matrix:")
print("              Predicted")
print("             0        1")
print(f"Actual 0:   {cmTN:6d}   {cmFP:6d}")
print(f"Actual 1:   {cmFN:6d}   {cmTP:6d}")

# Calculate Accuracy, TPR, and TNR
accuracy = 100 * (cmTP + cmTN) / total if total > 0 else 0
TPR = 100 * cmTP / (cmTP + cmFN) if (cmTP + cmFN) > 0 else 0
TNR = 100 * cmTN / (cmTN + cmFP) if (cmTN + cmFP) > 0 else 0

print(f"Accuracy: {accuracy:.2f}%")
print(f"True Positive Rate (Sensitivity): {TPR:.2f}%")
print(f"True Negative Rate (Specificity): {TNR:.2f}%")


basename = sys.argv[1] #file name and path from expLauncher
# Save the collected trial data to a text file with a timestamp
#output_file = os.path.join(output_folder, f"MCSE_s0{subject_number}_s0{subject_session}_r0{subject_run}_{timestamp}_set{set_size}_output.txt")
output_file = f"{basename}.output.txt"
with open(output_file, "w") as text_file:
    text_file.write(text_to_save)

#Save triggers and output together formatted for data analysis        
#analysis_file = os.path.join(output_folder, f"MCSE_s0{subject_number}_s0{subject_session}_r0{subject_run}_{timestamp}_set{set_size}_analysis.txt")
analysis_file = f"{basename}.analysis.txt"
with open(analysis_file, "w") as text_file:
    text_file.write(text_to_analyze)

# Save the triggers to a triggers.txt file
#trigger_file = os.path.join(output_folder, f"MCSE_s0{subject_number}_s0{subject_session}_r0{subject_run}_{timestamp}_set{set_size}_triggers.txt")
trigger_file = f"{basename}.triggers.txt"
with open(trigger_file, "w") as trigger_txt:
    for code, timestamp, trial in trigger:
        #formatted_trigger = f"[{code}]"
        trigger_txt.write(f"{trial+1} {code} {timestamp}\n")

# Stop the listener thread
listener_running[0] = False
listener_thread.join()
# Close BCI connections
bci.idStreamer_bus.Clear()
bci.iDsock_bus.close()
pygame.quit()
##### This one closes and saves the gdf file #####
# subprocess.run(["cl_rpc", "closexdf"])
sys.exit()