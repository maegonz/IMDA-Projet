import numpy as np
import pandas as pd
import torch
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from tqdm import tqdm
from IPython.display import HTML
from matplotlib.lines import Line2D


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
    return np.sin(rad), np.cos(rad)



INPUT_FILES = sorted(glob.glob("train/input_*.csv"))
DATASET_FILE = 'nfl_attention_dataset_CLEAN.pt' # Nouveau nom "CLEAN"
FUTURE_WINDOW = 10

data_list = []

print(f"📂 Traitement des fichiers : {INPUT_FILES}")

for input_file in INPUT_FILES:
    output_file = input_file.replace("input_", "output_")
    if not os.path.exists(output_file): continue

    print(f"🔄 Traitement {input_file}...")
    df_in = pd.read_csv(input_file)
    df_out = pd.read_csv(output_file)

    grouped = df_in.groupby(['game_id', 'play_id', 'frame_id'])

    for (game_id, play_id, frame_id), group in tqdm(grouped):
        # A. Receveur (Query)
        rec_row = group[group['player_role'] == 'Targeted Receiver']
        if len(rec_row) != 1: continue

        rx, ry = rec_row['x'].values[0], rec_row['y'].values[0]
        rec_id = rec_row['nfl_id'].values[0]
        play_dir = rec_row['play_direction'].values[0]

        # B. Label (Output)
        future_frame = frame_id + FUTURE_WINDOW
        future_pos = df_out[
            (df_out['game_id'] == game_id) &
            (df_out['play_id'] == play_id) &
            (df_out['nfl_id'] == rec_id) &
            (df_out['frame_id'] == future_frame)
        ]

        if len(future_pos) == 0: continue

        # Calcul du déplacement BRUT
        raw_dx = future_pos['x'].values[0] - rx
        raw_dy = future_pos['y'].values[0] - ry

        # CORRECTION CRUCIALE : Si le jeu est à gauche, on inverse le déplacement
        # Pour que "Avancer" soit toujours positif dans le référentiel du modèle
        if play_dir == 'left':
            label_dx = -raw_dx
            label_dy = -raw_dy
        else:
            label_dx = raw_dx
            label_dy = raw_dy

        label_tensor = torch.tensor([label_dx, label_dy], dtype=torch.float32)

        # C. Query Tensor (Standardisé)
        ball_x = rec_row['ball_land_x'].values[0]
        ball_y = rec_row['ball_land_y'].values[0]
        rec_s = rec_row['s'].values[0]
        rec_dir = rec_row['dir'].values[0]

        if play_dir == 'left':
            ball_x = 120 - ball_x
            ball_y = 53.3 - ball_y
            rx_std = 120 - rx
            ry_std = 53.3 - ry
            rec_dir = (rec_dir + 180) % 360
        else:
            rx_std = rx
            ry_std = ry
            # rec_dir reste tel quel

        q_feats = [rec_s, rec_dir, ball_x - rx_std, ball_y - ry_std]
        query_tensor = torch.tensor(q_feats, dtype=torch.float32)

        # D. Keys (Défenseurs Standardisés)
        defenders = group[group['player_side'] == 'Defense']
        if len(defenders) == 0: continue

        def_list = []
        for _, d in defenders.iterrows():
            dx, dy = d['x'], d['y']
            ds, ddir = d['s'], d['dir']

            if play_dir == 'left':
                dx = 120 - dx
                dy = 53.3 - dy
                ddir = (ddir + 180) % 360

            d_feats = [dx - rx_std, dy - ry_std, ds, ddir]
            def_list.append(d_feats)

        keys_tensor = torch.tensor(def_list, dtype=torch.float32)

        # Padding
        if keys_tensor.shape[0] < 11:
            padding = torch.zeros((11 - keys_tensor.shape[0], 4))
            keys_tensor = torch.cat([keys_tensor, padding], dim=0)
        else:
            keys_tensor = keys_tensor[:11]

        data_list.append((query_tensor, keys_tensor, label_tensor))

# --- SAUVEGARDE ---
print("\n📦 Sauvegarde...")
final_queries = torch.stack([x[0] for x in data_list])
final_keys = torch.stack([x[1] for x in data_list])
final_labels = torch.stack([x[2] for x in data_list])

dataset = {
    'queries': final_queries,
    'keys': final_keys,
    'labels': final_labels
}
torch.save(dataset, DATASET_FILE)
print("✅ Dataset CLEAN généré !")