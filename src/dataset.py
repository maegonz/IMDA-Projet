import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import glob
import os
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
# On cherche les fichiers dans le dossier nfl-big-data-bowl-2026-prediction puis dans train/
INPUT_FILES = glob.glob("nfl-big-data-bowl-2026-prediction/train/input_*.csv")
DATASET_FILE = 'nfl_dataset_v4_MULTI_AGENT.pt'
HISTORY_SIZE = 10 

# Normalisation
MAX_SPEED = 13.0
MAX_ACCEL = 10.0
MAX_DIST = 50.0

data_list = []

print(f"Génération du dataset : Multi-Agents (Tous les 'player_to_predict')...")

def get_angle_features(degree_val):
    rad = np.radians(degree_val)
    return np.sin(rad), np.cos(rad)

for in_file in INPUT_FILES:
    # Trouver le fichier output
    folder, filename = os.path.split(in_file)
    out_filename = filename.replace("input_", "output_")
    out_file = os.path.join(folder, out_filename)
    
    if not os.path.exists(out_file): continue
    
    df_in = pd.read_csv(in_file)
    df_out = pd.read_csv(out_file)
    
    common_plays = set(df_in['play_id']).intersection(df_out['play_id'])
    
    for play_id in tqdm(common_plays, leave=False, desc=filename):
        # Filtrer le play complet
        play_in = df_in[(df_in['game_id'] == df_in['game_id'].iloc[0]) & (df_in['play_id'] == play_id)].sort_values('frame_id')
        play_out = df_out[(df_out['game_id'] == df_out['game_id'].iloc[0]) & (df_out['play_id'] == play_id)].sort_values('frame_id')
        
        # On cherche TOUS les joueurs à prédire (pas juste le receveur visé)
        # On prend les IDs uniques qui sont marqués True à n'importe quel moment du play
        candidates = play_in[play_in['player_to_predict'] == True]['nfl_id'].unique()
        
        for target_id in candidates:
            # 1. Récupérer l'historique de CE joueur
            player_track = play_in[play_in['nfl_id'] == target_id]
            
            # S'il n'a pas assez d'historique (ex: vient de rentrer sur le terrain), on skip
            if len(player_track) < HISTORY_SIZE: continue
            
            # Séquence Input (les 10 dernières frames connues)
            history_seq = player_track.iloc[-HISTORY_SIZE:]
            current_pos = history_seq.iloc[-1]
            
            # 2. Récupérer son Futur (Label)
            future_track = play_out[play_out['nfl_id'] == target_id]
            if future_track.empty: continue # Pas de données futures dispos
            
            future_pos = future_track.iloc[-1]
            
            # Label = Offset (Déplacement relatif)
            label_x = (future_pos['x'] - current_pos['x'])
            label_y = (future_pos['y'] - current_pos['y'])
            label = torch.tensor([label_x, label_y], dtype=torch.float32)
            
            # --- CONSTRUCTION DES FEATURES ---
            
            # A. QUERY (Le joueur cible lui-même) : Séquence aplatie
            # Taille = 10 frames * 9 features = 90
            rec_features = []
            ball_x, ball_y = current_pos['ball_land_x'], current_pos['ball_land_y']
            
            for _, row in history_seq.iterrows():
                sin_d, cos_d = get_angle_features(row['dir'])
                sin_o, cos_o = get_angle_features(row['o'])
                
                # Est-ce que le joueur cible est en Attaque ?
                is_offense = 1.0 if play_in[play_in['nfl_id'] == target_id]['player_side'].iloc[0] == 'Offense' else 0.0

                feats = [
                    row['s'] / MAX_SPEED,
                    row['a'] / MAX_ACCEL,
                    sin_d, cos_d,
                    sin_o, cos_o,
                    (ball_x - row['x']) / MAX_DIST,
                    (ball_y - row['y']) / MAX_DIST,
                    is_offense
                ]
                rec_features.extend(feats)
            
            query = torch.tensor(rec_features, dtype=torch.float32)

            # B. KEYS (Le Contexte : TOUS les autres joueurs autour)
            # Important : On calcule la position relative par rapport au target_id actuel
            current_frame_id = current_pos['frame_id']
            
            # On prend tout le monde SAUF le joueur cible lui-même
            others = play_in[(play_in['frame_id'] == current_frame_id) & (play_in['nfl_id'] != target_id)]
            
            keys_list = []
            for _, other_row in others.iterrows():
                sin_d, cos_d = get_angle_features(other_row['dir'])
                sin_o, cos_o = get_angle_features(other_row['o'])
                
                # Est-ce un coéquipier ?
                # On compare le 'player_side' du voisin avec celui du target_id
                side_target = play_in[play_in['nfl_id'] == target_id]['player_side'].iloc[0]
                side_other = other_row['player_side']
                
                is_teammate = 1.0 if side_target == side_other else 0.0
                
                k_feats = [
                    (other_row['x'] - current_pos['x']) / MAX_DIST,
                    (other_row['y'] - current_pos['y']) / MAX_DIST,
                    other_row['s'] / MAX_SPEED,
                    other_row['a'] / MAX_ACCEL,
                    sin_d, cos_d,
                    sin_o, cos_o,
                    is_teammate # <--- NOUVELLE FEATURE
                ] 
                keys_list.append(k_feats)
            
            # Padding à 21 autres joueurs (11 def + 10 off max)
            # On augmente un peu la taille du padding pour être sûr
            while len(keys_list) < 22:
                keys_list.append([0.0] * 9)
                
            keys = torch.tensor(keys_list[:22], dtype=torch.float32) # (22, 9)

            data_list.append((query, keys, label))

            # 2. DATA AUGMENTATION : MIROIR VERTICAL (Flip Y)
            # L'axe Y va de 0 à 53.3. Inverser veut dire : y_new = 53.3 - y
            
            # Pour faire ça proprement sur les Tensors déjà créés :
            # Query (90 features) : [speed, accel, sin_d, cos_d, sin_o, cos_o, dist_x, dist_y] répété 10 fois
            # Inverser Y change le signe de : dist_y (feature 7) et cos_d/cos_o (composante verticale de l'angle)
            
            query_flip = query.clone()
            keys_flip = keys.clone()
            label_flip = label.clone()
            
            # Inverse la distance Y (Feature index 7, 15, 23... toutes les 8 features)
            # Note : Index 7 est dist_y. Index 3 est cos_d (direction Y). Index 5 est cos_o (orientation Y)
            
            # A. Inversion pour le Receveur (Query) - C'est une séquence aplatie
            # On inverse le signe de la composante Y (cosinus et distance Y)
            for i in range(10): # Pour les 10 frames
                base = i * 9
                query_flip[base + 3] *= -1 # Inverse Cos Direction (Y component)
                query_flip[base + 5] *= -1 # Inverse Cos Orientation (Y component)
                query_flip[base + 7] *= -1 # Inverse Dist Y
                
            # B. Inversion pour les Défenseurs (Keys) - (22, 9)
            # Features : [rel_x, rel_y, s, a, sin_d, cos_d, sin_o, cos_o]
            # Index à inverser : 1 (rel_y), 5 (cos_d), 7 (cos_o)
            keys_flip[:, 1] *= -1
            keys_flip[:, 5] *= -1
            keys_flip[:, 7] *= -1
            
            # C. Inversion du Label (x, y)
            label_flip[1] *= -1 # Inverse l'offset Y à prédire

            # Ajouter la version miroir
            data_list.append((query_flip, keys_flip, label_flip))

# Sauvegarde
print(f"Sauvegarde de {len(data_list)} exemples (Attaquants + Défenseurs)...")
if len(data_list) > 0:
    final_queries = torch.stack([x[0] for x in data_list])
    final_keys = torch.stack([x[1] for x in data_list])
    final_labels = torch.stack([x[2] for x in data_list])
    
    torch.save({'queries': final_queries, 'keys': final_keys, 'labels': final_labels}, DATASET_FILE)
    print(f"Dataset prêt : {DATASET_FILE}")
else:
    print("Aucun exemple généré.")