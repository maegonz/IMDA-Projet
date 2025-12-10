import numpy as np
import pandas as pd
import torch
import glob
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

INPUT_FILES = sorted(glob.glob("./train/input_*.csv"))
FUTURE_WINDOW = 10  # Position prediction in 10 frames (1 second)
columns_dropped = [
    'player_height', 'player_weight', 'player_birth_date', 'player_name'
]
columns_groupedby = [
    'game_id', 'play_id', 'frame_id'
]

data_list = []
limit_x, limit_y = 120, 53.3  # Field dimensions in yards
    
class NFLDataset(Dataset):
    def __init__(self,
                 input_dir: str = './data/train/'):
        
        # list of input files paths
        self.list_inputs = sorted(glob.glob(os.path.join(input_dir, "input_*.csv")))

        # data = torch.load(pt_file)
        # self.queries = data['queries']
        # self.keys = data['keys']
        # self.labels = data['labels']


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_file = self.list_inputs[idx]
        output_file = input_file.replace("input_", "output_")

        assert os.path.exists(output_file), f"Output file not found for {input_file}"

        print(f"🔄 Processing {input_file}...")
        df_in, df_out = pd.read_csv(input_file), pd.read_csv(output_file)
        print(f" - Input shape : {df_in.shape}, Output shape : {df_out.shape}")

        grouped = df_in.groupby(columns_groupedby)

        for (game_id, play_id, frame_id), group in tqdm(grouped):

            # A. Receiver (Query)
            rec_row = group[group['player_role'] == 'Targeted Receiver']
            if len(rec_row) != 1: continue

            # receiver position
            rx, ry = rec_row['x'].values[0], rec_row['y'].values[0]
            # receiver id
            rec_id = rec_row['nfl_id'].values[0]
            # player's direction LEFT or RIGHT
            play_dir = rec_row['play_direction'].values[0]
            # receiver's speed and direction
            rec_s = rec_row['s'].values[0]  # speed in yards/sec
            rec_dir = rec_row['dir'].values[0]  # direction in degrees

            # B. Label (Output)
            future_frame = frame_id + FUTURE_WINDOW
            future_pos = df_out[
                (df_out['game_id'] == game_id) &
                (df_out['play_id'] == play_id) &
                (df_out['nfl_id'] == rec_id) &
                (df_out['frame_id'] == future_frame)
            ]

            if len(future_pos) == 0: continue

            # Movement brut
            raw_dx = future_pos['x'].values[0] - rx
            raw_dy = future_pos['y'].values[0] - ry

            # CRUCIAL CORRECTION: If the play is going left, invert the movement
            # So that "Moving forward" is always positive in the model's reference frame
            if play_dir == 'left':
                label_dx = -raw_dx
                label_dy = -raw_dy
            else:
                label_dx = raw_dx
                label_dy = raw_dy

            label_tensor = torch.tensor([label_dx, label_dy], dtype=torch.float32)

            # C. Query Tensor (Standardisé)
            # ball landing position
            ball_x, ball_y = rec_row['ball_land_x'].values[0], rec_row['ball_land_y'].values[0]

            if play_dir == 'left':
                ball_x = limit_x - ball_x
                ball_y = limit_y - ball_y
                rx_std = limit_x - rx
                ry_std = limit_y - ry
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
                    dx = limit_x - dx
                    dy = limit_y - dy
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

        q = self.queries[idx].clone()
        k = self.keys[idx].clone()
        y = self.labels[idx].clone()

        # --- NORMALISATION ---
        # Query: [Speed, Dir, DistX, DistY]
        q[0] /= 10.0   # Speed (max ~10)
        q[1] /= 360.0  # Dir
        q[2] /= 50.0   # DistX (max ~50)
        q[3] /= 50.0   # DistY

        # Keys: [RelX, RelY, Speed, Dir]
        k[:, 0] /= 50.0
        k[:, 1] /= 50.0
        k[:, 2] /= 10.0
        k[:, 3] /= 360.0

        # Label: [dX, dY] (Déplacement sur 10 frames, env 10 yards max)
        y /= 10.0

        return q, k, y
    
    def _movement_correction(self, dx, dy, play_dir):
        if play_dir == 'left':
            return limit_x - dx, limit_y - dy
        return dx, dy