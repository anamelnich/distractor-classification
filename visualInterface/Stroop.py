import pygame
import random
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import json
import time
import sys
import queue
import os 
from datetime import datetime 
import heapq
from python_client import Trigger

#Part 1 evaluates reading ability
#Part 2 provides baseline for RT analysis

pygame.init()

screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
SCREEN_WIDTH, SCREEN_HEIGHT = screen.get_size()

pygame.display.set_caption("Stroop Task")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

FONT_SIZE = 72
font = pygame.font.SysFont('Times New Roman', FONT_SIZE)

SAMPLE_RATE = 16000 #Hz
audio_queue = queue.Queue()

# Initialize Vosk model
try:
    model = Model("./vosk/vosk-model-small-en-us-0.15")

    # vocabulary = '["red", "green", "blue", "yellow"]'
    # recognizer = KaldiRecognizer(model, SAMPLE_RATE, vocabulary)

except Exception as e:
    print("Error initializing Vosk model:", e)
    sys.exit(1)

class StroopTask:
    def __init__(self):
        self.colors = ['RED', 'GREEN', 'BLUE', 'YELLOW']
        self.color_values = {
            'RED': RED,
            'GREEN': GREEN,
            'BLUE': BLUE,
            'YELLOW': YELLOW
        }
        self.trials_data = []
        self.current_trial = 0
        self.recording = False
        self.response_time = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.basename = sys.argv[1]
        self.log_filename = f"{self.basename}.behoutput.txt"
        self.parallel = Trigger('USB2LPT')
        self.parallel.init(50)
        self.model = model
            
    def generate_part1_trials(self):
        trials = []
        for _ in range(6):
            temp = self.colors.copy()
            random.shuffle(temp)
            trials.extend(temp)
        return trials
        
    def generate_part2_trials(self):
        trials = []
        for color in self.colors:
            trials.extend([color] * 6)
        random.shuffle(trials)
        return trials
        
    def generate_part3_trials(self):
        """
        Generate 60 trials for part 3 that are randomly ordered by trial type (20 congruent, 20 incongruent, 20 neutral)
        and assign an ink color to each trial (15 occurrences per color) such that no two consecutive trials share the same ink color.
        
        For each trial:
        - Congruent: the displayed word is the same as the ink color.
        - Incongruent: the displayed word is a random color (from the three not equal to the ink).
        - Neutral: the displayed word is chosen from a list of neutral words.
        """
        neutral_words = ["COUCH", "DRESS", "BRIDGE", "DOG"]
        
        # 1. Create and shuffle trial types.
        trial_types = ['congruent'] * 20 + ['incongruent'] * 20 + ['neutral'] * 20
        random.shuffle(trial_types)
        
        # 2. Backtracking algorithm to assign ink colors.
        def backtrack(sequence, counts, last_color):
            if len(sequence) == 60:
                return sequence
            # Randomize the order in which we try colors.
            colors = self.colors.copy()
            random.shuffle(colors)
            for color in colors:
                if color == last_color:
                    continue  # avoid consecutive same color
                if counts[color] > 0:
                    counts[color] -= 1
                    sequence.append(color)
                    result = backtrack(sequence, counts, color)
                    if result is not None:
                        return result
                    sequence.pop()
                    counts[color] += 1
            return None

        counts = {color: 15 for color in self.colors}
        ink_list = backtrack([], counts, None)
        if ink_list is None:
            print("ERROR: Could not generate ink list with backtracking.")
            sys.exit(1)
        
        # 3. Combine trial types and ink colors into a trial list.
        trials = []
        for i in range(60):
            trial_type = trial_types[i]
            ink = ink_list[i]
            if trial_type == 'congruent':
                word = ink
            elif trial_type == 'incongruent':
                # Choose a word randomly from the colors that are not equal to the ink.
                possible_words = [c for c in self.colors if c != ink]
                word = random.choice(possible_words)
            else:  # neutral
                word = random.choice(neutral_words)
            trials.append((trial_type, word, ink))
        return trials
    def generate_part4_trials(self):
        """
        Practice for part 3
        """
        neutral_words = ["COUCH", "DRESS", "BRIDGE", "DOG"]
        
        # 1. Create and shuffle trial types.
        trial_types = ['congruent'] * 8 + ['incongruent'] * 8 + ['neutral'] * 8
        random.shuffle(trial_types)
        
        # 2. Backtracking algorithm to assign ink colors.
        def backtrack(sequence, counts, last_color):
            if len(sequence) == 24:
                return sequence
            # Randomize the order in which we try colors.
            colors = self.colors.copy()
            random.shuffle(colors)
            for color in colors:
                if color == last_color:
                    continue  # avoid consecutive same color
                if counts[color] > 0:
                    counts[color] -= 1
                    sequence.append(color)
                    result = backtrack(sequence, counts, color)
                    if result is not None:
                        return result
                    sequence.pop()
                    counts[color] += 1
            return None

        counts = {color: 6 for color in self.colors}
        ink_list = backtrack([], counts, None)
        if ink_list is None:
            print("ERROR: Could not generate ink list with backtracking.")
            sys.exit(1)
        
        # 3. Combine trial types and ink colors into a trial list.
        trials = []
        for i in range(24):
            trial_type = trial_types[i]
            ink = ink_list[i]
            if trial_type == 'congruent':
                word = ink
            elif trial_type == 'incongruent':
                # Choose a word randomly from the colors that are not equal to the ink.
                possible_words = [c for c in self.colors if c != ink]
                word = random.choice(possible_words)
            else:  # neutral
                word = random.choice(neutral_words)
            trials.append((trial_type, word, ink))
        return trials


        
    def display_stimulus(self, text, color=WHITE, is_circle=False):
        screen.fill(BLACK)
        if is_circle:
            pygame.draw.circle(screen, color, (SCREEN_WIDTH//2, SCREEN_HEIGHT//2), 50)
        else:
            text_surface = font.render(text, True, color)
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            screen.blit(text_surface, text_rect)
        pygame.display.flip()
        
    def display_fixation(self):
        screen.fill(BLACK)
        text_surface = font.render("+", True, WHITE)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
        screen.blit(text_surface, text_rect)
        pygame.display.flip()
        
    def run_trial(self, stimulus, is_circle=False, ink_color=None, trial_type=None):

        recognizer = KaldiRecognizer(self.model, SAMPLE_RATE, '["red", "green", "blue", "yellow"]')
        recognizer.SetWords(True)

        self.display_fixation()
        self.parallel.signal(6)

        pygame.time.wait(500)
        
        # Display the stimulus (for text or circle)
        if is_circle:
            self.display_stimulus("", self.color_values[stimulus], True)
            # self.parallel.signal(1)
        else:
            color = self.color_values[ink_color] if ink_color else WHITE
            self.display_stimulus(stimulus, color)
            if trial_type == 'congruent':
                self.parallel.signal(10)
            elif trial_type == 'incongruent':
                self.parallel.signal(20)
            elif trial_type == 'neutral':
                self.parallel.signal(30)
        
        # Record stimulus onset and capture audio (recording for 2 seconds here)
        stimulus_time = time.time()
        duration = 2  # seconds
        audio_data = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished
        audio_bytes = audio_data.tobytes()
        
        # Process the recorded audio using Vosk
        recognizer.SetWords(True)
        if recognizer.AcceptWaveform(audio_bytes):
            result_json = recognizer.Result()
        else:
            result_json = recognizer.FinalResult()
        print("Recognizer returned result:", result_json)
        result = json.loads(result_json)
        
        attempts = []
        correct_response = None
        reaction_time = None
        
        if "text" in result and result["text"]:
            recognized_text = result["text"].upper().strip()
            if recognized_text == "READ":
                recognized_text = "RED"
            if "result" in result and len(result["result"]) > 0:
                first_word = result["result"][0]
                reaction_time = first_word["start"] * 1000
            else:
                reaction_time = None
            attempts.append((recognized_text, reaction_time))
            if ink_color is None:
                if recognized_text == stimulus:
                    correct_response = recognized_text
            else:
                if recognized_text == ink_color:
                    correct_response = recognized_text
        else:
            print("No speech recognized.")
        
        timeout = (len(attempts) == 0)
        correct = (correct_response is not None)
        
        print(f"Trial ended. Attempts: {attempts}, Timeout: {timeout}, Reaction Time: {reaction_time}")
        
        trial_data = {
            "trial_type": trial_type,
            "stimulus": stimulus,
            "ink_color": ink_color if ink_color else "NA",
            "response": 3 if timeout else (1 if correct else 2),
            "reaction_time": reaction_time if reaction_time is not None else "NA",
            "correct": correct,
            "timeout": timeout,
            "num_attempts": len(attempts),
            "attempts": attempts
        }
        
        self.trials_data.append(trial_data)
        
        # Provide feedback to the participant
        screen.fill(BLACK)
        feedback = "timeout" if timeout else ("correct" if correct else "incorrect")
        text_surface = font.render(feedback, True, WHITE)
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
        screen.blit(text_surface, text_rect)
        pygame.display.flip()
        pygame.time.wait(500)
        
    def save_data(self, part_number):
        # Create results directory if it doesn't exist
        # if not os.path.exists('results'):
        #     os.makedirs('results')
            
        # Create filename with timestamp and part number
        # filename = f'results/stroop_part{part_number}_{self.timestamp}.txt'
        
        with open(self.log_filename, 'w') as f:
            # Write header 
            f.write("Trial\tTrial_Type\tStimulus\tInk_Color\tResponse\tReaction_Time\n")
            
            # Write data for each trial
            for i, trial in enumerate(self.trials_data, 1):
                rt = f"{trial['reaction_time']:.3f}" if isinstance(trial['reaction_time'], float) else trial['reaction_time']
                f.write(f"{i}\t"
                        f"{trial['trial_type']}\t"
                        f"{trial['stimulus']}\t"
                        f"{trial['ink_color'] if trial['ink_color'] else 'N/A'}\t"
                        f"{trial['response']}\t"
                        f"{rt}\n")
                        

        
    def run_part(self, part):
        # Clear previous trials data
        self.trials_data = []
        
        if part == 1:
            trials = self.generate_part1_trials()
            for trial in trials:
                self.run_trial(trial)
                
        elif part == 2:
            trials = self.generate_part2_trials()
            for trial in trials:
                self.run_trial(trial, is_circle=True)
                
        elif part == 3:
            trials = self.generate_part3_trials()
            
            # Display a message to start the run
            screen.fill(BLACK)
            text = font.render("Press SPACE to begin run", True, WHITE)
            screen.blit(text, text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)))
            pygame.display.flip()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        waiting = False
            
            # Run through the 60 trials for part 3
            for trial in trials:
                trial_type, word, color = trial
                self.run_trial(word, ink_color=color, trial_type=trial_type)
        elif part == 4:
            trials = self.generate_part4_trials()
            
            # Display a message to start the run
            screen.fill(BLACK)
            text = font.render("Press SPACE to begin run", True, WHITE)
            screen.blit(text, text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)))
            pygame.display.flip()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        waiting = False
            
            # Run through the 60 trials for part 3
            for trial in trials:
                trial_type, word, color = trial
                self.run_trial(word, ink_color=color, trial_type=trial_type)
        
        # Save data after part is completed
        self.save_data(part)

# Create task instance
stroop_task = StroopTask()

# Main menu loop
running = True
while running:
    screen.fill(BLACK)
    text = font.render("Select Part (1, 2, 3, or 4) or Q to quit", True, WHITE)
    screen.blit(text, text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2)))
    pygame.display.flip()
    
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_1:
                print("Starting Part 1")
                stroop_task.run_part(1)
            elif event.key == pygame.K_2:
                print("Starting Part 2")
                stroop_task.run_part(2)
            elif event.key == pygame.K_3:
                print("Starting Part 3")
                stroop_task.run_part(3)
            elif event.key == pygame.K_4:
                print("Starting Part 4")
                stroop_task.run_part(4)
            elif event.key == pygame.K_q:
                running = False

pygame.quit()

