import numpy as np
from typing import Union


# --- CONSTANTES ---
HISTORY_SIZE = 10
MAX_SPEED = 13.0
MAX_ACCEL = 10.0
MAX_DIST = 50.0
NUM_FRAMES = 10
MAX_ANGLE = 360.0

def get_angle_features(degree_val):
    rad = np.radians(degree_val)
    sin, cos = np.sin(rad), np.cos(rad)
    return sin, cos

def normalisation(queries, keys, labels=None):
        # --- NORMALISATION ---
        # Query: [Speed, Dir, DistX, DistY]
        queries[0] /= MAX_SPEED   # Speed (max ~10)
        queries[1] /= MAX_ANGLE  # Dir
        queries[2] /= MAX_DIST   # DistX (max ~50)
        queries[3] /= MAX_DIST   # DistY

        # Keys: [RelX, RelY, Speed, Dir]
        keys[:, 0] /= MAX_DIST
        keys[:, 1] /= MAX_DIST
        keys[:, 2] /= MAX_SPEED
        keys[:, 3] /= MAX_ANGLE

        if labels is not None:
            # Label: [dX, dY] (Déplacement sur 10 frames, env 10 yards max)
            labels /= NUM_FRAMES # 10 frames = 1 seconde

        return queries, keys, labels

def limit(x, y):
    limit_x, limit_y = 120, 53.3  # Field dimensions in yards

    x = limit_x - x
    y = limit_y - y
    return x, y

def movement_correction(dx: float, 
                        dy: float, 
                        play_direction: bool, 
                        direction: Union[float, None]=None,  
                        ball=None):
    
    if ball is not None:
        if play_direction == 'left':
            # Ball's position correction
            dx, dy = limit(dx, dy)
            return dx, dy

    if play_direction == 'left':
        dx, dy = limit(dx, dy)
        direction = (direction + 180) % MAX_ANGLE
        return dx, dy, direction
    return dx, dy, direction