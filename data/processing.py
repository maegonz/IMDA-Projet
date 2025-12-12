import numpy as np
import pandas as pd
import glob
import os
from typing import Union


# --- CONSTANTES ---
HISTORY_SIZE = 10
MAX_SPEED = 13.0
MAX_ACCEL = 10.0
MAX_DIST = 50.0
NUM_FRAMES = 10
MAX_ANGLE = 360.0

def load_all_data(folder_path):
    # 1. Charger tous les inputs
    input_files = glob.glob(os.path.join(folder_path, "input_*.csv"))
    print(f"Fichiers Input trouvés : {len(input_files)}")
    df_in_list = [pd.read_csv(f) for f in input_files]
    if df_in_list:
        df_in_all = pd.concat(df_in_list, ignore_index=True)
    else:
        print("⚠️ Aucun fichier input trouvé !")
        return None, None

    # 2. Charger tous les outputs
    output_files = glob.glob(os.path.join(folder_path, "output_*.csv"))
    print(f"Fichiers Output trouvés : {len(output_files)}")
    df_out_list = [pd.read_csv(f) for f in output_files]
    if df_out_list:
        df_out_all = pd.concat(df_out_list, ignore_index=True)
    else:
        df_out_all = pd.DataFrame() # Vide

    return df_in_all, df_out_all


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