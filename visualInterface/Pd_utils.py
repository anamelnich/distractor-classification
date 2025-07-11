# utils.py
import math
import random
import pygame
from typing import NamedTuple
import Pd_config as config
from Pd_logger import get_current_logger

# ================== Display Utilities ==================

class Display(NamedTuple):
    screen:           pygame.Surface
    screen_width_px:  int
    screen_height_px: int
    pixels_per_cm:    float
    x_center:         float
    y_center:         float
    d_from_center:    float

def init_display():
    """
    Initialize Pygame fullscreen and compute all of your
    visual‐angle → pixel conversions in one shot.
    """
    pygame.init()
    screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
    sw_px, sh_px = screen.get_size()

    # cm → pixels scaling
    ppcm_x = sw_px / config.screen_width_cm
    ppcm_y = sh_px / config.screen_height_cm
    ppcm   = (ppcm_x + ppcm_y) / 2 # pixels per cm

    # screen center
    x_ctr = sw_px / 2
    y_ctr = sh_px / 2

    # eccentricity in px
    d_px = degrees_to_pixels(
        config.eccentricity_deg,
        config.viewing_distance_cm,
        ppcm
    )

    return Display(
        screen=screen,
        screen_width_px=sw_px,
        screen_height_px=sh_px,
        pixels_per_cm=ppcm,
        x_center=x_ctr,
        y_center=y_ctr,
        d_from_center=d_px
    )
def degrees_to_pixels(degrees, viewing_distance_cm, pixels_per_cm):
    radians = math.radians(degrees)
    size_cm = 2 * viewing_distance_cm * math.tan(radians / 2)
    size_px = size_cm * pixels_per_cm
    return size_px

# ================== Shape Utilities ==================

def draw_circle(screen, x, y, radius, color=(0,128,0)):     # circle is always green, never a distractor
    pygame.draw.circle(screen, color, (int(x), int(y)), int(radius))

def draw_square(screen, x, y, width, height, color):
    rect = (int(x-width/2), int(y-height/2), int(width), int(height))
    pygame.draw.rect(screen, color, rect)

def draw_diamond(screen, x, y, width, height, color):
    pts = [
        (int(x), int(y-height/2)),
        (int(x+width/2), int(y)),
        (int(x), int(y+height/2)),
        (int(x-width/2), int(y))
    ]
    pygame.draw.polygon(screen, color, pts)

def draw_hexagon(screen, x, y, width, height, color):
    radius = width / 2
    pts = [
        (x + radius * math.cos(math.radians(60 * i)),
         y + radius * math.sin(math.radians(60 * i)))
        for i in range(6)
    ]
    pts = [(int(px), int(py)) for px, py in pts]
    pygame.draw.polygon(screen, color, pts)


def draw_dot(screen, x, y, shape_width, side):
    dot_radius = config.dot_radius
    offset = shape_width * 0.25
    dot_x = x + (offset if side else -offset)
    dot_y = y
    pygame.draw.circle(screen, (0,0,0), (int(dot_x), int(dot_y)), dot_radius)

def compute_shape_definitions_px(ppcm: float):
    """
    Take your config.shape_definitions (in degrees)
    and convert each entry to a pixel‐based dict.
    """
    px_defs = []
    for shape in config.shape_definitions:
        if shape["type"] == "circle":
            diam = degrees_to_pixels(shape["size_deg"],
                                     config.viewing_distance_cm,
                                     ppcm)
            px_defs.append({
                "type":       "circle",
                "diameter_px": diam,
                "radius_px":   diam / 2
            })
        else:
            w_deg, h_deg = shape["size_deg"]
            px_defs.append({
                "type":    shape["type"],
                "width_px":  degrees_to_pixels(w_deg,
                                               config.viewing_distance_cm,
                                               ppcm),
                "height_px": degrees_to_pixels(h_deg,
                                               config.viewing_distance_cm,
                                               ppcm)
            })
    return px_defs

def compute_shape_coords(set_size: int, d_from_center: float,
                         x_center: float, y_center: float):
    """
    Return a list of (x,y) screen coordinates
    evenly spaced on a circle of radius d_from_center.
    """
    coords = []
    offset = math.pi / 2  # start at top
    for i in range(set_size):
        angle = i * (2*math.pi / set_size) - offset
        x = x_center + d_from_center * math.cos(angle)
        y = y_center + d_from_center * math.sin(angle)
        coords.append((x, y))
    return coords

# ================== Trial Randomization Utilities ==================

def compute_positions_for_set_size(set_size):
    """Returns (mid_pos, lat_pos, left_pos) for your given set_size."""
    if set_size == 4:
        return [1,3], [2,4], [2,4]
    if set_size == 6:
        return [1,4], [2,3,5,6], [2,3]
    if set_size == 8:
        return [1,5], [2,3,4,6,7,8], [6,7,8]
    if set_size == 10:
        return [1,6], list(range(2,6))+list(range(7,11)), list(range(6,11))
    raise ValueError(f"Unsupported set_size: {set_size}")

def generate_trials(n_trials, set_size):
    """
    Returns four lists:
      trial_types:      length n_trials, 0=no-distractor or 1=distractor
      d_pos:            distractor position (0 if none)
      t_pos:            target position (1..set_size)
      shape_positions:  list of dicts mapping pos->(shapeName, dotSide)
    """
    # 1) trial types
    half = n_trials // 2
    trial_types = [0]*half + [1]*half
    random.shuffle(trial_types)
    # avoid >3 repeats
    while any(trial_types[i:i+4] == [trial_types[i]]*4 
              for i in range(n_trials-3)):
        random.shuffle(trial_types)

    mid_pos, lat_pos, left_pos = compute_positions_for_set_size(set_size)

    # 2) distractor positions
    d_pos = [0]*n_trials
    n_d = trial_types.count(1)
    pool = left_pos * math.ceil(n_d/len(left_pos))
    random.shuffle(pool)
    for i, tt in enumerate(trial_types):
        if tt == 1:
            d_pos[i] = pool.pop(0)

    # 3) target positions
    t_pos = [0]*n_trials
    half_d = n_d // 2
    pool_t = mid_pos * half_d
    random.shuffle(pool_t)
    for i, tt in enumerate(trial_types):
        if tt == 1:
            t_pos[i] = pool_t.pop(0)
        else:
            t_pos[i] = random.randint(1, set_size)

    # 4) Shape + dot-side assignment
    base_shapes = ['Diamond','Hexagon','Square'] * ((set_size-1)//3 + 1)
    shape_positions = []
    for i, tt in enumerate(trial_types):
        trial_dict = {}

        # Circle at target, with its own dot side
        circle_side = random.choice([0, 1])
        trial_dict[t_pos[i]] = ('Circle', circle_side)

        # The other shapes
        available = [p for p in range(1, set_size+1) if p != t_pos[i]]
        random.shuffle(available)
        shape_names = base_shapes[: set_size-1]
        for pos, name in zip(available, shape_names):
            dot_side = random.choice([0, 1])
            trial_dict[pos] = (name, dot_side)

        shape_positions.append(trial_dict)

    return trial_types, d_pos, t_pos, shape_positions

# ================== Trigger Utilities ==================

def init_hardware_trigger():
    global HWTrigger
    from python_client import Trigger
    HWTrigger = Trigger(config.trigger_typ)
    HWTrigger.init(50)


def add_trigger(code, trial_index=None):
    ts = pygame.time.get_ticks()
    if config.MODE in ('train','decode'):
        HWTrigger.signal(code)
    logger = get_current_logger() 
    if logger is not None:
        logger.log_trigger(code, ts, trial_index)
