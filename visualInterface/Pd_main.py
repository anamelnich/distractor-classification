# main.py
import sys
import os
import argparse
import random
import pygame
from datetime import datetime
import threading
from cnbiloop import BCI_tid

import Pd_config as config
import Pd_utils as utils
from Pd_logger import TrialLogger, set_current_logger


def send_tid(value):
    bci.id_msg_bus.SetEvent(value)
    bci.iDsock_bus.sendall(str.encode(bci.id_serializer_bus.Serialize()))

def receiveTiD():
    global bci
    data = None
    try:
        data = bci.iDsock_bus.recv(512).decode("utf-8")
        bci.idStreamer_bus.Append(data)
    except BlockingIOError as e:
        if e.errno != 11:
            print("BlockingIOError in receiveTiD:", e)
    except Exception as e:
        print("Error in receiveTiD:", e)
    if data:
        if bci.idStreamer_bus.Has("<tobiid", "/>"):
            msg = bci.idStreamer_bus.Extract("<tobiid", "/>")
            bci.id_serializer_bus.Deserialize(msg)
            bci.idStreamer_bus.Clear()
            return int(round(float(bci.id_msg_bus.GetEvent())))
        elif bci.idStreamer_bus.Has("<tcstatus", "/>"):
            # drain unwanted messages
            count = bci.idStreamer_bus.Count("<tcstatus")
            for _ in range(1, count-1):
                bci.idStreamer_bus.Extract("<tcstatus", "/>")
    return None

def trigger_listener(running_flag):
    global correct_detected, error_detected, uncertainty_detected
    while running_flag[0]:
        msg = receiveTiD()
        if msg:
            if msg == 2:

                correct_detected = True
            elif msg == 1:

                error_detected = True
            elif msg == 3:

                uncertainty_detected = True
        pygame.time.wait(20)

def run_training_mode(basename):
    print("Running in TRAINING mode...")
    # --- Controller init ---
    pygame.joystick.init()
    js = None
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        print("Joystick connected:", js.get_name())
    else:
        print("No joystick detected")
    disp = utils.init_display()
    screen        = disp.screen
    pixels_per_cm = disp.pixels_per_cm
    x_center      = disp.x_center
    y_center      = disp.y_center
    d_from_center = disp.d_from_center
    
    pygame.font.init()
    font = pygame.font.SysFont(config.font, config.font_size)

    # Convert shape sizes → px
    shape_px_defs = utils.compute_shape_definitions_px(pixels_per_cm)
    shape_def_map = {d["type"].lower(): d for d in shape_px_defs}
    set_size      = len(shape_px_defs)
    # Compute on‐screen positions
    shape_coords  = utils.compute_shape_coords(
        set_size      = len(shape_px_defs),
        d_from_center = d_from_center,
        x_center      = x_center,
        y_center      = y_center
    )

    log = TrialLogger(basename)
    set_current_logger(log) 

    trial_type, d_pos, t_pos, shape_positions = utils.generate_trials(
        config.n_trials,
        set_size
    )
    mid_pos, lat_pos, left_pos, right_pos = utils.compute_positions_for_set_size(set_size)
    if config.MODE in ("train","decode"):
        utils.init_hardware_trigger()
    
    responses = []
    trial_idx = 0
    run            = True

    pygame.mouse.set_visible(False)

    while run and trial_idx < config.n_trials:
        # global quit (ESC)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                run = False
        if not run:
            break
        
        # start screen on first trial
        if trial_idx == 0:
            screen.fill((0,0,0))
            text = font.render("Press any key to start", True, (255,255,255))
            rect = text.get_rect(center=(x_center,y_center))
            screen.blit(text, rect)
            pygame.display.flip()
            # wait for a key press
            waiting = True
            while waiting:
                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN or ev.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False
        
        if trial_idx == config.break_trial:
            screen.fill((0,0,0))
            text = font.render("Time for a break", True, (255,255,255))
            rect = text.get_rect(center=(x_center,y_center))
            screen.blit(text, rect)
            pygame.display.flip()
            pygame.time.delay(config.break_duration_ms) # 20 seconds mandatory

            screen.fill((0,0,0))
            text = font.render("Press any key to start", True, (255,255,255))
            rect = text.get_rect(center=(x_center,y_center))
            screen.blit(text, rect)
            pygame.display.flip()
            # wait for a key press
            pygame.event.clear()
            waiting = True
            while waiting:
                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN or ev.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False
        
        task      = trial_type[trial_idx]
        tpos      = t_pos[trial_idx]
        dpos      = d_pos[trial_idx]
        shape_map = shape_positions[trial_idx]
        t_side = 1 if tpos in lat_pos else 0
        d_side = 1 if dpos in right_pos else (2 if dpos in left_pos else 0)


        draw_map = {
            "circle":   utils.draw_circle,
            "square":   utils.draw_square,
            "diamond":  utils.draw_diamond,
            "hexagon":  utils.draw_hexagon
        }

        #Draw blank screen
        screen.fill((0, 0, 0))
        pygame.display.update()
        ITI = utils.get_random_delay(config.delay_duration_ms)
        pygame.time.delay(ITI)
   
        # Draw fixation cross
        pygame.draw.line(screen, (225, 225, 225), (x_center - config.fixation_size, y_center), (x_center + config.fixation_size, y_center), config.line_width)
        pygame.draw.line(screen, (225, 225, 225), (x_center, y_center - config.fixation_size), (x_center, y_center + config.fixation_size), config.line_width)
        pygame.display.update()
        utils.add_trigger(4, trial_idx)
        pygame.time.delay(config.fixation_duration_ms)

        array_start_time = pygame.time.get_ticks()
        trial_end = False
        resp_recorded   = False
        # start_trigger = int(f"1{t_side}{d_pos[trial_idx]}")
        if d_side==0:
            start_trigger = 8
        elif d_side==1:
            start_trigger=32
        elif d_side==2:
            start_trigger=44
        utils.add_trigger(start_trigger, trial_idx)

        while not trial_end:
            now = pygame.time.get_ticks()

            # time‐out check
            if now - array_start_time >= config.stimulus_duration_ms:
                trial_end = True
                if not resp_recorded:               # no response → timeout
                    response     = 3
                    utils.add_trigger(64, trial_idx)
            else:
                # screen.fill((0,0,0))

                for pos, (shape_name, dot_side) in shape_positions[trial_idx].items():
                    x, y = shape_coords[pos-1]
                    sdef = shape_def_map[shape_name.lower()]   # your pixel‐size dict

                    color = (255,0,0) if (trial_type[trial_idx]==1 and d_pos[trial_idx]==pos) else (0,128,0)

                    if shape_name.lower() == "circle":
                        # for circles we need radius
                        utils.draw_circle(screen, x, y, sdef["radius_px"], color)
                        shape_width = sdef["diameter_px"]
                        dot_correct = dot_side
                    else:
                        # use our map to call the correct function
                        fn = draw_map[shape_name.lower()]
                        fn(screen, x, y, sdef["width_px"], sdef["height_px"], color)
                        shape_width = sdef["width_px"]

                    utils.draw_dot(screen, x, y, shape_width, dot_side)

                pygame.display.flip()


                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        trial_end = True
                        run       = False
                        break

                    if not resp_recorded:

                        recorded_this_event = False

                        # Mouse clicks
                        if ev.type == pygame.MOUSEBUTTONDOWN:
                            if ev.button == 1:          # left click
                                trial_end = True
                                resp_recorded = True
                                is_correct = (dot_correct == 0)
                                recorded_this_event = True

                            elif ev.button == 3:        # right click
                                trial_end = True
                                resp_recorded = True
                                is_correct = (dot_correct == 1)
                                recorded_this_event = True

                            else:
                                continue  # ignore other mouse buttons

                        # Controller bumpers
                        elif ev.type == pygame.JOYBUTTONDOWN:
                            # print("BUTTON", ev.button)  # debug if you want
                            if ev.button == 4:          # LB
                                trial_end = True
                                resp_recorded = True
                                is_correct = (dot_correct == 0)
                                recorded_this_event = True

                            elif ev.button == 5:        # RB
                                trial_end = True
                                resp_recorded = True
                                is_correct = (dot_correct == 1)
                                recorded_this_event = True

                            else:
                                continue  # ignore other controller buttons

                        else:
                            continue  # not an input event we care about

                        # Only compute response if we actually recorded one
                        if recorded_this_event:
                            response     = 1 if is_correct else 2
                            trigger_code = 11 if is_correct else 12
                            utils.add_trigger(64, trial_idx)
                            break
        screen.fill((0, 0, 0))
        if response == 1:
            response_text = 'Correct'
        elif response == 2:
            response_text = 'Incorrect'
        else:  # response == 3
            response_text = 'Timeout'

        feedback_surf = font.render(response_text, True, (225, 225, 225))
        feedback_rect = feedback_surf.get_rect(center=(x_center, y_center))
        screen.blit(feedback_surf, feedback_rect)
        pygame.display.flip()
        pygame.time.delay(config.feedback_duration_ms)
        responses.append(response)

        log.log_trial( trial_idx, task, response, tpos, dpos, dot_correct, ITI)
        trial_idx += 1

    # --- SUMMARY SCREEN ---

    correct_pct   = responses.count(1) / config.n_trials
    incorrect_pct = responses.count(2) / config.n_trials
    timeout_pct   = responses.count(3) / config.n_trials

    # draw text
    screen.fill((0, 0, 0))
    summary_str = (
        f"Correct:   {correct_pct:.2f}    "
        f"Incorrect: {incorrect_pct:.2f}    "
        f"Timeout:   {timeout_pct:.2f}"
    )
    surf = font.render(summary_str, True, (225, 225, 225))
    rect = surf.get_rect(center=(x_center, y_center))
    screen.blit(surf, rect)
    pygame.display.flip()

    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN:
                waiting = False

    log.dump()
    pygame.quit()


def run_decoding_mode(basename):
    print("Running in DECODING mode...")
    # --- Controller init ---
    pygame.joystick.init()
    js = None
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        print("Joystick connected:", js.get_name())
    else:
        print("No joystick detected")
    # 1) exactly same display + font + shape setup as training
    disp = utils.init_display()
    screen        = disp.screen
    pixels_per_cm = disp.pixels_per_cm
    x_center      = disp.x_center
    y_center      = disp.y_center
    d_from_center = disp.d_from_center

    pygame.font.init()
    font = pygame.font.SysFont(config.font, config.font_size)

    shape_px_defs = utils.compute_shape_definitions_px(pixels_per_cm)
    shape_def_map = {d["type"].lower(): d for d in shape_px_defs}
    set_size      = len(shape_px_defs)
    shape_coords  = utils.compute_shape_coords(set_size,
                                               d_from_center,
                                               x_center,
                                               y_center)

    log = TrialLogger(basename)
    set_current_logger(log)

    trial_type, d_pos, t_pos, shape_positions = utils.generate_trials(
        config.n_trials,
        set_size
    )
    mid_pos, lat_pos, left_pos, right_pos = utils.compute_positions_for_set_size(set_size)

    # init hardware triggers & BCI listener
    utils.init_hardware_trigger()
    global bci, correct_detected, error_detected, uncertainty_detected
    bci = BCI_tid.BciInterface()
    correct_detected = False
    error_detected   = False
    uncertainty_detected = False

    listener_running = [True]
    listener_thread  = threading.Thread(target=trigger_listener,
                                        args=(listener_running,))
    listener_thread.daemon = True
    listener_thread.start()

    # load feedback icons
    icon_size  = config.icon_size
    thumb_up   = pygame.transform.scale(
                    pygame.image.load("./img/thumb_up.png").convert_alpha(),
                    icon_size)
    thumb_down = pygame.transform.scale(
                    pygame.image.load("./img/thumb_down.png").convert_alpha(),
                    icon_size)
    uncertain = pygame.transform.scale(
                pygame.image.load("./img/line.png").convert_alpha(),
                icon_size)

    # performance counters
    cmTP = cmFP = cmTN = cmFN = 0
    cmTPr = cmTPl = cmFNr = cmFNl = 0

    score = 0
    uncertain_count=0

    responses = []
    BCI_output = []
    trial_idx = 0
    run       = True
    pygame.mouse.set_visible(False)
    # 5) main trial loop
    while run and trial_idx < config.n_trials:
        # ESC to abort & notify BCI
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                run = False
        if not run:
            break

        # first‐trial start screen
        if trial_idx == 0:
            screen.fill((0,0,0))
            t = font.render("Press any key to start", True, (255,255,255))
            r = t.get_rect(center=(x_center,y_center))
            screen.blit(t, r); pygame.display.flip()
            waiting = True
            while waiting:
                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN or ev.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False
        if trial_idx == config.break_trial:
            screen.fill((0,0,0))
            text = font.render("Time for a break", True, (255,255,255))
            rect = text.get_rect(center=(x_center,y_center))
            screen.blit(text, rect)
            pygame.display.flip()
            pygame.time.delay(5000) 

            screen.fill((0,0,0))
            text = font.render("Press any key to start", True, (255,255,255))
            rect = text.get_rect(center=(x_center,y_center))
            screen.blit(text, rect)
            pygame.display.flip()
            # wait for a key press
            pygame.event.clear()
            waiting = True
            while waiting:
                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN or ev.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False
        # unpack trial params
        task      = trial_type[trial_idx]
        tpos      = t_pos[trial_idx]
        dpos      = d_pos[trial_idx]
        shape_map = shape_positions[trial_idx]
        t_side    = 1 if tpos in lat_pos else 0
        d_side    = 1 if dpos in right_pos else (2 if dpos in left_pos else 0)

        draw_map = {
            "square":  utils.draw_square,
            "diamond": utils.draw_diamond,
            "hexagon": utils.draw_hexagon
        }

        # blank + delay
        screen.fill((0,0,0)); pygame.display.update()
        ITI = utils.get_random_delay(config.delay_duration_ms)
        pygame.time.delay(ITI)

        # fixation
        pygame.draw.line(screen, (225,225,225),
                         (x_center-config.fixation_size, y_center),
                         (x_center+config.fixation_size, y_center),
                         config.line_width)
        pygame.draw.line(screen, (225,225,225),
                         (x_center, y_center-config.fixation_size),
                         (x_center, y_center+config.fixation_size),
                         config.line_width)
        pygame.display.update()
        utils.add_trigger(4, trial_idx)
        pygame.time.delay(config.fixation_duration_ms)

        # stimulus + send start trigger
        array_start = pygame.time.get_ticks()
        trial_end   = False
        resp_recorded = False
        # start_tr    = int(f"1{t_side}{dpos}")
        if d_side==0:
            start_tr = 8
        elif d_side==1:
            start_tr=32
        elif d_side==2:
            start_tr=44
        utils.add_trigger(start_tr, trial_idx)
        wait = False
        while not trial_end:
            now = pygame.time.get_ticks()
            if now - array_start >= config.stimulus_duration_ms:
                trial_end = True
                if not resp_recorded:
                    utils.add_trigger(64, trial_idx)
                    response = 3
            elif wait:
                screen.fill((0, 0, 0))
                pygame.display.update()
            else:
                # screen.fill((0,0,0))
                for pos, (shape_name, dot_side) in shape_map.items():
                    x, y  = shape_coords[pos-1]
                    sdef  = shape_def_map[shape_name.lower()]
                    is_d  = (task==1 and dpos==pos)
                    color = (255,0,0) if is_d else (0,128,0)

                    if shape_name.lower()=="circle":
                        utils.draw_circle(screen, x, y,
                                          sdef["radius_px"], color)
                        shape_width = sdef["diameter_px"]
                        dot_correct = dot_side
                    else:
                        fn = draw_map[shape_name.lower()]
                        fn(screen, x, y,
                           sdef["width_px"], sdef["height_px"], color)
                        shape_width = sdef["width_px"]

                    utils.draw_dot(screen, x, y, shape_width, dot_side)
                pygame.display.flip()

                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        trial_end = True
                        run       = False
                        break

                    if not resp_recorded:

                        recorded_this_event = False

                        # Mouse clicks
                        if ev.type == pygame.MOUSEBUTTONDOWN:
                            if ev.button == 1:          # left click
                                # trial_end = True
                                resp_recorded = True
                                wait=True
                                is_correct = (dot_correct == 0)
                                recorded_this_event = True

                            elif ev.button == 3:        # right click
                                # trial_end = True
                                resp_recorded = True
                                wait=True
                                is_correct = (dot_correct == 1)
                                recorded_this_event = True

                            else:
                                continue  # ignore other mouse buttons

                        # Controller bumpers
                        elif ev.type == pygame.JOYBUTTONDOWN:
                            # print("BUTTON", ev.button)  # debug if you want
                            if ev.button == 4:          # LB
                                # trial_end = True
                                resp_recorded = True
                                wait=True
                                is_correct = (dot_correct == 0)
                                recorded_this_event = True

                            elif ev.button == 5:        # RB
                                # trial_end = True
                                resp_recorded = True
                                wait=True
                                is_correct = (dot_correct == 1)
                                recorded_this_event = True

                            else:
                                continue  # ignore other controller buttons

                        else:
                            continue  # not an input event we care about

                        # Only compute response if we actually recorded one
                        if recorded_this_event:
                            response     = 1 if is_correct else 2
                            trigger_code = 11 if is_correct else 12
                            utils.add_trigger(64, trial_idx)
                            # break

        # 6) BCI‐driven feedback
        screen.fill((0,0,0))
        icon = None
        if correct_detected:
            Pd_class = 1
            if task == 1:
                response_text = "Score: +2"; response_color = (0,255,0)
                score+=2
                icon = thumb_up; cmTP += 1; 
                if d_side == 1:
                    cmTPr += 1
                elif d_side == 2:
                    cmTPl += 1
            else:
                response_text = "Score: -1"; response_color = (255,0,0)
                score-=1
                icon = thumb_down; cmFP += 1; 
            correct_detected = False

        elif error_detected:
            Pd_class = 0
            if task == 1:
                response_text = "Score: -1"; response_color = (255,0,0)
                score-=1
                icon = thumb_down; cmFN += 1; 
                if d_side == 1:
                    cmFNr += 1
                elif d_side == 2:
                    cmFNl += 1
            else:
                response_text = "Score: +1"; response_color = (0,255,0)
                score+=1
                icon = thumb_up; cmTN += 1; 

            error_detected = False
        elif uncertainty_detected:
            Pd_class = 3
            response_text = "Score: NA"; response_color = (255,255,255)
            icon = uncertain
            uncertainty_detected = False
            uncertain_count+=1
        else:
            Pd_class = 4
            response_text = "No Model Output"
            response_color = (255,255,255)


        if icon is not None:
            icon_rect = icon.get_rect(center=(x_center, y_center))
            screen.blit(icon, icon_rect)
            # now we know icon_rect exists:
            text_surf = font.render(response_text, True, response_color)
            text_rect = text_surf.get_rect(midtop=(x_center, icon_rect.bottom + 10))
            screen.blit(text_surf, text_rect)
        else:
            # no icon: just center the text vertically instead
            text_surf = font.render(response_text, True, response_color)
            text_rect = text_surf.get_rect(center=(x_center, y_center))
            screen.blit(text_surf, text_rect)
        pygame.display.update()
        pygame.time.delay(1000)

        responses.append(response)
        BCI_output.append(Pd_class)
        log.log_trial(trial_idx, task, response,
                      tpos, dpos, dot_correct, ITI,Pd_class)

        trial_idx += 1

    correct_pct   = responses.count(1) / config.n_trials
    incorrect_pct = responses.count(2) / config.n_trials
    timeout_pct   = responses.count(3) / config.n_trials

    total = cmTP + cmFP + cmFN + cmTN
    accuracy = 100*(cmTP+cmTN)/total if total>0 else 0
    TPR = 100*cmTP/(cmTP+cmFN) if (cmTP+cmFN)>0 else 0
    TNR = 100*cmTN/(cmTN+cmFP) if (cmTN+cmFP)>0 else 0
    prct_uncertain = 100*uncertain_count/config.n_trials
    TPRr = 100*cmTPr/(cmTPr+cmFNr) if (cmTPr+cmFNr)>0 else 0
    TPRl = 100*cmTPl/(cmTPl+cmFNl) if (cmTPl+cmFNl)>0 else 0

    screen.fill((0,0,0))
    utils.show_final_screen(screen, x_center, y_center, accuracy, TPRr, TPRl, TNR, config)
    # utils.show_final_screen(screen, x_center, y_center, 74, 65, 71, 71, config)
    # big_font = pygame.font.SysFont(config.font, 80)   # size 80 for big
    # score_surf = big_font.render(f"Score: {score} out of 90", True, (255,255,255))
    # score_rect = score_surf.get_rect(center=(x_center, y_center - 40))
    # screen.blit(score_surf, score_rect)

    # small_font = pygame.font.SysFont(config.font, 32)  # size 32 for smaller text
    # metrics_y = score_rect.bottom + 20                 # 20px below big score

    # metrics_lines = [
    #     f"Correct:   {correct_pct:.2f}",
    #     f"Incorrect: {incorrect_pct:.2f}",
    #     f"Timeout:   {timeout_pct:.2f}"
    # ]
    # for i, line in enumerate(metrics_lines):
    #     m_surf = small_font.render(line, True, (200,200,200))
    #     m_rect = m_surf.get_rect(center=(x_center, metrics_y + i*40))
    #     screen.blit(m_surf, m_rect)

    pygame.display.flip()

    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN:
                waiting = False
    send_tid(20)

    print("Confusion Matrix:")
    print("              Predicted")
    print("             0        1")
    print(f"Actual 0:   {cmTN:6d}   {cmFP:6d}")
    print(f"Actual 1:   {cmFN:6d}   {cmTP:6d}")

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"TPR:      {TPR:.2f}%")
    print(f"TNR:      {TNR:.2f}%")
    print(f"% uncertain: {prct_uncertain:.2f}")

    print(f"TPR right distractor:      {TPRr:.2f}%")
    print(f"TPR left distractor:       {TPRl:.2f}%")

    utils.update_threshold_instructions(TPRr, TPRl, TNR, prct_uncertain, accuracy)

    listener_running[0] = False
    listener_thread.join()
    bci.idStreamer_bus.Clear()
    bci.iDsock_bus.close()

    log.dump()
    pygame.quit()

def run_decoding_ctrl_mode(basename):
    print("Running in DECODING mode...")
    # --- Controller init ---
    pygame.joystick.init()
    js = None
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        print("Joystick connected:", js.get_name())
    else:
        print("No joystick detected")
    # 1) exactly same display + font + shape setup as training
    disp = utils.init_display()
    screen        = disp.screen
    pixels_per_cm = disp.pixels_per_cm
    x_center      = disp.x_center
    y_center      = disp.y_center
    d_from_center = disp.d_from_center

    pygame.font.init()
    font = pygame.font.SysFont(config.font, config.font_size)

    shape_px_defs = utils.compute_shape_definitions_px(pixels_per_cm)
    shape_def_map = {d["type"].lower(): d for d in shape_px_defs}
    set_size      = len(shape_px_defs)
    shape_coords  = utils.compute_shape_coords(set_size,
                                               d_from_center,
                                               x_center,
                                               y_center)

    log = TrialLogger(basename)
    set_current_logger(log)

    trial_type, d_pos, t_pos, shape_positions = utils.generate_trials(
        config.n_trials,
        set_size
    )
    mid_pos, lat_pos, left_pos, right_pos = utils.compute_positions_for_set_size(set_size)

    # init hardware triggers & BCI listener
    utils.init_hardware_trigger()
    global bci, correct_detected, error_detected, uncertainty_detected
    bci = BCI_tid.BciInterface()
    correct_detected = False
    error_detected   = False
    uncertainty_detected = False

    listener_running = [True]
    listener_thread  = threading.Thread(target=trigger_listener,
                                        args=(listener_running,))
    listener_thread.daemon = True
    listener_thread.start()

    # load feedback icons
    icon_size  = config.icon_size
    thumb_up   = pygame.transform.scale(
                    pygame.image.load("./img/thumb_up.png").convert_alpha(),
                    icon_size)
    thumb_down = pygame.transform.scale(
                    pygame.image.load("./img/thumb_down.png").convert_alpha(),
                    icon_size)
    uncertain = pygame.transform.scale(
                pygame.image.load("./img/line.png").convert_alpha(),
                icon_size)

    # performance counters
    cmTP = cmFP = cmTN = cmFN = 0
    cmTPr = cmTPl = cmFNr = cmFNl = 0

    score = 0
    uncertain_count=0

    responses = []
    BCI_output = []
    trial_idx = 0
    run       = True
    pygame.mouse.set_visible(False)

    # 5) main trial loop
    while run and trial_idx < config.n_trials:
        # ESC to abort & notify BCI
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                run = False
        if not run:
            break

        # first‐trial start screen
        if trial_idx == 0:
            screen.fill((0,0,0))
            t = font.render("Press any key to start", True, (255,255,255))
            r = t.get_rect(center=(x_center,y_center))
            screen.blit(t, r); pygame.display.flip()
            waiting = True
            while waiting:
                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN or ev.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False
        if trial_idx == config.break_trial:
            screen.fill((0,0,0))
            text = font.render("Time for a break", True, (255,255,255))
            rect = text.get_rect(center=(x_center,y_center))
            screen.blit(text, rect)
            pygame.display.flip()
            pygame.time.delay(5000) 

            screen.fill((0,0,0))
            text = font.render("Press any key to start", True, (255,255,255))
            rect = text.get_rect(center=(x_center,y_center))
            screen.blit(text, rect)
            pygame.display.flip()
            # wait for a key press
            pygame.event.clear()
            waiting = True
            while waiting:
                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN or ev.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False
        # unpack trial params
        task      = trial_type[trial_idx]
        tpos      = t_pos[trial_idx]
        dpos      = d_pos[trial_idx]
        shape_map = shape_positions[trial_idx]
        t_side    = 1 if tpos in lat_pos else 0
        d_side    = 1 if dpos in right_pos else (2 if dpos in left_pos else 0)

        draw_map = {
            "square":  utils.draw_square,
            "diamond": utils.draw_diamond,
            "hexagon": utils.draw_hexagon
        }

        # blank + delay
        screen.fill((0,0,0)); pygame.display.update()
        ITI = utils.get_random_delay(config.delay_duration_ms)
        pygame.time.delay(ITI)

        # fixation
        pygame.draw.line(screen, (225,225,225),
                         (x_center-config.fixation_size, y_center),
                         (x_center+config.fixation_size, y_center),
                         config.line_width)
        pygame.draw.line(screen, (225,225,225),
                         (x_center, y_center-config.fixation_size),
                         (x_center, y_center+config.fixation_size),
                         config.line_width)
        pygame.display.update()
        utils.add_trigger(4, trial_idx)
        pygame.time.delay(config.fixation_duration_ms)

        # stimulus + send start trigger
        array_start = pygame.time.get_ticks()
        trial_end   = False
        resp_recorded = False
        # start_tr    = int(f"1{t_side}{dpos}")
        if d_side==0:
            start_tr = 8
        elif d_side==1:
            start_tr=32
        elif d_side==2:
            start_tr=44
        utils.add_trigger(start_tr, trial_idx)
        wait = False
        while not trial_end:
            now = pygame.time.get_ticks()
            if now - array_start >= config.stimulus_duration_ms:
                trial_end = True
                if not resp_recorded:
                    utils.add_trigger(64, trial_idx)
                    response = 3
            elif wait:
                screen.fill((0, 0, 0))
                pygame.display.update()
            else:
                # screen.fill((0,0,0))
                for pos, (shape_name, dot_side) in shape_map.items():
                    x, y  = shape_coords[pos-1]
                    sdef  = shape_def_map[shape_name.lower()]
                    is_d  = (task==1 and dpos==pos)
                    color = (255,0,0) if is_d else (0,128,0)

                    if shape_name.lower()=="circle":
                        utils.draw_circle(screen, x, y,
                                          sdef["radius_px"], color)
                        shape_width = sdef["diameter_px"]
                        dot_correct = dot_side
                    else:
                        fn = draw_map[shape_name.lower()]
                        fn(screen, x, y,
                           sdef["width_px"], sdef["height_px"], color)
                        shape_width = sdef["width_px"]

                    utils.draw_dot(screen, x, y, shape_width, dot_side)
                pygame.display.flip()
                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        trial_end = True
                        run       = False
                        break

                    if not resp_recorded:

                        recorded_this_event = False

                        # Mouse clicks
                        if ev.type == pygame.MOUSEBUTTONDOWN:
                            if ev.button == 1:          # left click
                                # trial_end = True
                                resp_recorded = True
                                wait=True
                                is_correct = (dot_correct == 0)
                                recorded_this_event = True

                            elif ev.button == 3:        # right click
                                # trial_end = True
                                resp_recorded = True
                                wait=True
                                is_correct = (dot_correct == 1)
                                recorded_this_event = True

                            else:
                                continue  # ignore other mouse buttons

                        # Controller bumpers
                        elif ev.type == pygame.JOYBUTTONDOWN:
                            # print("BUTTON", ev.button)  # debug if you want
                            if ev.button == 4:          # LB
                                # trial_end = True
                                resp_recorded = True
                                wait=True
                                is_correct = (dot_correct == 0)
                                recorded_this_event = True

                            elif ev.button == 5:        # RB
                                # trial_end = True
                                resp_recorded = True
                                wait=True
                                is_correct = (dot_correct == 1)
                                recorded_this_event = True

                            else:
                                continue  # ignore other controller buttons

                        else:
                            continue  # not an input event we care about

                        # Only compute response if we actually recorded one
                        if recorded_this_event:
                            response     = 1 if is_correct else 2
                            trigger_code = 11 if is_correct else 12
                            utils.add_trigger(64, trial_idx)
                            # break

        # 6) BCI‐driven feedback
        screen.fill((0,0,0))
        icon = None
        if correct_detected:
            Pd_class = 1
            if task == 1:
                response_text = "Score: +2"; response_color = (0,255,0)
                score+=2
                icon = thumb_up; cmTP += 1; 
                if d_side == 1:
                    cmTPr += 1
                elif d_side == 2:
                    cmTPl += 1
            else:
                response_text = "Score: -1"; response_color = (255,0,0)
                score-=1
                icon = thumb_down; cmFP += 1; 
            correct_detected = False

        elif error_detected:
            Pd_class = 0
            if task == 1:
                response_text = "Score: -1"; response_color = (255,0,0)
                score-=1
                icon = thumb_down; cmFN += 1; 
                if d_side == 1:
                    cmFNr += 1
                elif d_side == 2:
                    cmFNl += 1
            else:
                response_text = "Score: +1"; response_color = (0,255,0)
                score+=1
                icon = thumb_up; cmTN += 1; 

            error_detected = False
        elif uncertainty_detected:
            Pd_class = 3
            response_text = "Score: NA"; response_color = (255,255,255)
            icon = uncertain
            uncertainty_detected = False
            uncertain_count+=1
        else:
            Pd_class = 4
            response_text = "No Model Output"
            response_color = (255,255,255)


        if icon is not None:
            icon_rect = icon.get_rect(center=(x_center, y_center))
            screen.blit(icon, icon_rect)
            # now we know icon_rect exists:
            text_surf = font.render(response_text, True, response_color)
            text_rect = text_surf.get_rect(midtop=(x_center, icon_rect.bottom + 10))
            screen.blit(text_surf, text_rect)
        else:
            # no icon: just center the text vertically instead
            text_surf = font.render(response_text, True, response_color)
            text_rect = text_surf.get_rect(center=(x_center, y_center))
            screen.blit(text_surf, text_rect)
        # pygame.display.update()
        pygame.time.delay(1000)

        responses.append(response)
        BCI_output.append(Pd_class)
        log.log_trial(trial_idx, task, response,
                      tpos, dpos, dot_correct, ITI,Pd_class)

        trial_idx += 1

    correct_pct   = responses.count(1) / config.n_trials
    incorrect_pct = responses.count(2) / config.n_trials
    timeout_pct   = responses.count(3) / config.n_trials

    total = cmTP + cmFP + cmFN + cmTN
    accuracy = 100*(cmTP+cmTN)/total if total>0 else 0
    TPR = 100*cmTP/(cmTP+cmFN) if (cmTP+cmFN)>0 else 0
    TNR = 100*cmTN/(cmTN+cmFP) if (cmTN+cmFP)>0 else 0
    prct_uncertain = 100*uncertain_count/config.n_trials
    TPRr = 100*cmTPr/(cmTPr+cmFNr) if (cmTPr+cmFNr)>0 else 0
    TPRl = 100*cmTPl/(cmTPl+cmFNl) if (cmTPl+cmFNl)>0 else 0

    screen.fill((0,0,0))
    # utils.show_final_screen(screen, x_center, y_center, accuracy, TPRr, TPRl, TNR, config)

    screen.fill((0, 0, 0))
    summary_str = (
        f"Correct:   {correct_pct:.2f}    "
        f"Incorrect: {incorrect_pct:.2f}    "
        f"Timeout:   {timeout_pct:.2f}"
    )
    surf = font.render(summary_str, True, (225, 225, 225))
    rect = surf.get_rect(center=(x_center, y_center))
    screen.blit(surf, rect)
    pygame.display.flip()

    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN:
                waiting = False

    send_tid(20)

    # print("Confusion Matrix:")
    # print("              Predicted")
    # print("             0        1")
    # print(f"Actual 0:   {cmTN:6d}   {cmFP:6d}")
    # print(f"Actual 1:   {cmFN:6d}   {cmTP:6d}")

    # print(f"Accuracy: {accuracy:.2f}%")
    # print(f"TPR:      {TPR:.2f}%")
    # print(f"TNR:      {TNR:.2f}%")
    # print(f"% uncertain: {prct_uncertain:.2f}")

    # print(f"TPR right distractor:      {TPRr:.2f}%")
    # print(f"TPR left distractor:       {TPRl:.2f}%")

    utils.update_threshold_instructions(TPRr, TPRl, TNR, prct_uncertain, accuracy)

    listener_running[0] = False
    listener_thread.join()
    bci.idStreamer_bus.Clear()
    bci.iDsock_bus.close()

    log.dump()
    pygame.quit()
def run_test_mode(basename):
    print("Running in TESTING mode...")
    # --- Controller init ---
    pygame.joystick.init()
    js = None
    if pygame.joystick.get_count() > 0:
        js = pygame.joystick.Joystick(0)
        js.init()
        print("Joystick connected:", js.get_name())
    else:
        print("No joystick detected")
    disp = utils.init_display()
    screen        = disp.screen
    pixels_per_cm = disp.pixels_per_cm
    x_center      = disp.x_center
    y_center      = disp.y_center
    d_from_center = disp.d_from_center
    
    pygame.font.init()
    font = pygame.font.SysFont(config.font, config.font_size)

    # Convert shape sizes → px
    shape_px_defs = utils.compute_shape_definitions_px(pixels_per_cm)
    shape_def_map = {d["type"].lower(): d for d in shape_px_defs}
    set_size      = len(shape_px_defs)
    # Compute on‐screen positions
    shape_coords  = utils.compute_shape_coords(
        set_size      = len(shape_px_defs),
        d_from_center = d_from_center,
        x_center      = x_center,
        y_center      = y_center
    )

    log = TrialLogger(basename)
    set_current_logger(log) 

    trial_type, d_pos, t_pos, shape_positions = utils.generate_trials(
        config.n_trials,
        set_size
    )
    mid_pos, lat_pos, left_pos, right_pos = utils.compute_positions_for_set_size(set_size)
    
    responses = []
    trial_idx = 0
    run            = True
    pygame.mouse.set_visible(False)
    
    while run and trial_idx < config.n_trials:
        # global quit (ESC)
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                run = False
        if not run:
            break
        
        # start screen on first trial
        if trial_idx == 0:
            screen.fill((0,0,0))
            text = font.render("Press any key to start", True, (255,255,255))
            rect = text.get_rect(center=(x_center,y_center))
            screen.blit(text, rect)
            pygame.display.flip()
            # wait for a key press
            waiting = True
            while waiting:
                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN or ev.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False
        
        if trial_idx == config.break_trial:
            screen.fill((0,0,0))
            text = font.render("Time for a break", True, (255,255,255))
            rect = text.get_rect(center=(x_center,y_center))
            screen.blit(text, rect)
            pygame.display.flip()
            pygame.time.delay(config.break_duration_ms) # 20 seconds mandatory

            screen.fill((0,0,0))
            text = font.render("Press any key to start", True, (255,255,255))
            rect = text.get_rect(center=(x_center,y_center))
            screen.blit(text, rect)
            pygame.display.flip()
            # wait for a key press
            pygame.event.clear()
            waiting = True
            while waiting:
                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN or ev.type == pygame.MOUSEBUTTONDOWN:
                        waiting = False
        
        task      = trial_type[trial_idx]
        tpos      = t_pos[trial_idx]
        dpos      = d_pos[trial_idx]
        shape_map = shape_positions[trial_idx]
        t_side = 1 if tpos in lat_pos else 0
        d_side    = 1 if dpos in right_pos else (2 if dpos in left_pos else 0)
        draw_map = {
            "circle":   utils.draw_circle,
            "square":   utils.draw_square,
            "diamond":  utils.draw_diamond,
            "hexagon":  utils.draw_hexagon
        }

        #Draw blank screen
        screen.fill((0, 0, 0))
        pygame.display.update()
        ITI = utils.get_random_delay(config.delay_duration_ms)
        pygame.time.delay(ITI)
   
        # Draw fixation cross
        pygame.draw.line(screen, (225, 225, 225), (x_center - config.fixation_size, y_center), (x_center + config.fixation_size, y_center), config.line_width)
        pygame.draw.line(screen, (225, 225, 225), (x_center, y_center - config.fixation_size), (x_center, y_center + config.fixation_size), config.line_width)
        pygame.display.update()
        utils.add_trigger(4, trial_idx)
        pygame.time.delay(config.fixation_duration_ms)

        array_start_time = pygame.time.get_ticks()
        trial_end = False
        resp_recorded   = False
        # start_trigger = int(f"1{t_side}{d_pos[trial_idx]}")
        if d_side==0:
            start_trigger = 8
        elif d_side==1:
            start_trigger=32
        elif d_side==2:
            start_trigger=44
        utils.add_trigger(start_trigger, trial_idx)

        while not trial_end:
            now = pygame.time.get_ticks()

            # time‐out check
            if now - array_start_time >= config.stimulus_duration_ms:
                trial_end = True
                if not resp_recorded:               # no response → timeout
                    response     = 3
                    utils.add_trigger(64, trial_idx)
            else:
                # screen.fill((0,0,0))

                for pos, (shape_name, dot_side) in shape_positions[trial_idx].items():
                    x, y = shape_coords[pos-1]
                    sdef = shape_def_map[shape_name.lower()]   # your pixel‐size dict

                    color = (255,0,0) if (trial_type[trial_idx]==1 and d_pos[trial_idx]==pos) else (0,128,0)

                    if shape_name.lower() == "circle":
                        # for circles we need radius
                        utils.draw_circle(screen, x, y, sdef["radius_px"], color)
                        shape_width = sdef["diameter_px"]
                        dot_correct = dot_side
                    else:
                        # use our map to call the correct function
                        fn = draw_map[shape_name.lower()]
                        fn(screen, x, y, sdef["width_px"], sdef["height_px"], color)
                        shape_width = sdef["width_px"]

                    utils.draw_dot(screen, x, y, shape_width, dot_side)

                pygame.display.flip()

                for ev in pygame.event.get():
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                        trial_end = True
                        run       = False
                        break

                    if not resp_recorded:

                        recorded_this_event = False

                        # Mouse clicks
                        if ev.type == pygame.MOUSEBUTTONDOWN:
                            if ev.button == 1:          # left click
                                trial_end = True
                                resp_recorded = True
                                is_correct = (dot_correct == 0)
                                recorded_this_event = True

                            elif ev.button == 3:        # right click
                                trial_end = True
                                resp_recorded = True
                                is_correct = (dot_correct == 1)
                                recorded_this_event = True

                            else:
                                continue  # ignore other mouse buttons

                        # Controller bumpers
                        elif ev.type == pygame.JOYBUTTONDOWN:
                            # print("BUTTON", ev.button)  # debug if you want
                            if ev.button == 4:          # LB
                                trial_end = True
                                resp_recorded = True
                                is_correct = (dot_correct == 0)
                                recorded_this_event = True

                            elif ev.button == 5:        # RB
                                trial_end = True
                                resp_recorded = True
                                is_correct = (dot_correct == 1)
                                recorded_this_event = True

                            else:
                                continue  # ignore other controller buttons

                        else:
                            continue  # not an input event we care about

                        # Only compute response if we actually recorded one
                        if recorded_this_event:
                            response     = 1 if is_correct else 2
                            trigger_code = 11 if is_correct else 12
                            utils.add_trigger(64, trial_idx)
                            break
        screen.fill((0, 0, 0))
        if response == 1:
            response_text = 'Correct'
        elif response == 2:
            response_text = 'Incorrect'
        else:  # response == 3
            response_text = 'Timeout'

        feedback_surf = font.render(response_text, True, (225, 225, 225))
        feedback_rect = feedback_surf.get_rect(center=(x_center, y_center))
        screen.blit(feedback_surf, feedback_rect)
        pygame.display.flip()
        pygame.time.delay(config.feedback_duration_ms)
        responses.append(response)

        log.log_trial( trial_idx, task, response, tpos, dpos, dot_correct, ITI)
        trial_idx += 1

    # --- SUMMARY SCREEN ---

    correct_pct   = responses.count(1) / config.n_trials
    incorrect_pct = responses.count(2) / config.n_trials
    timeout_pct   = responses.count(3) / config.n_trials

    # draw text
    screen.fill((0, 0, 0))
    summary_str = (
        f"Correct:   {correct_pct:.2f}    "
        f"Incorrect: {incorrect_pct:.2f}    "
        f"Timeout:   {timeout_pct:.2f}"
    )
    surf = font.render(summary_str, True, (225, 225, 225))
    rect = surf.get_rect(center=(x_center, y_center))
    screen.blit(surf, rect)
    pygame.display.flip()

    waiting = True
    while waiting:
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN:
                waiting = False

    log.dump()
    pygame.quit()

def main():

    if len(sys.argv) == 2:
        config.MODE = "test"
        basename = datetime.now().strftime("test_%Y%m%d%H%M%S")
    else:
        basename, mode = sys.argv[1], sys.argv[2].lower()
        config.MODE = mode
        os.makedirs(os.path.dirname(basename), exist_ok=True)

    if config.MODE == "train":
        run_training_mode(basename)
    elif config.MODE == "decode":
        run_decoding_mode(basename)
    elif config.MODE == "decode_ctrl":
        run_decoding_ctrl_mode(basename)
    else:
        run_test_mode(basename)

if __name__ == "__main__":
    main()