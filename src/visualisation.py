import pandas as pd
import glob
import os
from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from matplotlib.lines import Line2D

# Mettez le chemin vers votre dossier contenant les CSV
# Exemple : ""
path_to_train = "../nfl-big-data-bowl-2026-prediction/train/" 

# --- CONSTANTES (Doivent être identiques à l'entraînement) ---
HISTORY_SIZE = 10
MAX_SPEED = 13.0
MAX_ACCEL = 10.0
MAX_DIST = 50.0

def get_angle_features(degree_val):
    rad = np.radians(degree_val)
    return np.sin(rad), np.cos(rad)

# --- 1. FONCTION DE PRÉPARATION V4 (Cruciale) ---
def prepare_input_v4(df, game_id, play_id, target_nfl_id):
    """
    Prépare les inputs complexes (Historique 10 frames + Physique + Drapeaux)
    pour un joueur spécifique.
    """
    # Filtrer le play complet
    play_data = df[(df['game_id'] == game_id) & (df['play_id'] == play_id)].sort_values('frame_id')
    
    # 1. Récupérer l'historique du joueur cible
    player_track = play_data[play_data['nfl_id'] == target_nfl_id]
    
    # Vérification : a-t-on assez d'historique ?
    if len(player_track) < HISTORY_SIZE:
        return None, None, None
        
    # Séquence Input (les 10 dernières frames)
    history_seq = player_track.iloc[-HISTORY_SIZE:]
    current_pos = history_seq.iloc[-1] # La dernière frame (T)
    
    # --- A. CONSTRUCTION DU RECEIVER (QUERY) ---
    # Taille attendue : 90 (10 frames * 9 features)
    rec_features = []
    ball_x, ball_y = current_pos['ball_land_x'], current_pos['ball_land_y']
    
    # Est-ce un attaquant ?
    is_offense = 1.0 if current_pos['player_side'] == 'Offense' else 0.0
    
    for _, row in history_seq.iterrows():
        sin_d, cos_d = get_angle_features(row['dir'])
        sin_o, cos_o = get_angle_features(row['o'])
        
        feats = [
            row['s'] / MAX_SPEED,
            row['a'] / MAX_ACCEL,
            sin_d, cos_d,
            sin_o, cos_o,
            (ball_x - row['x']) / MAX_DIST,
            (ball_y - row['y']) / MAX_DIST,
            is_offense # Feature 9 : Ami ou Ennemi (pour soi-même)
        ]
        rec_features.extend(feats)
        
    query = torch.tensor(rec_features, dtype=torch.float32).unsqueeze(0) # (1, 90)

    # --- B. CONSTRUCTION DES DÉFENSEURS/PARTENAIRES (KEYS) ---
    # On prend le contexte à la dernière frame connue
    current_frame_id = current_pos['frame_id']
    others = play_data[(play_data['frame_id'] == current_frame_id) & (play_data['nfl_id'] != target_nfl_id)]
    
    keys_list = []
    side_target = current_pos['player_side']
    
    for _, other_row in others.iterrows():
        sin_d, cos_d = get_angle_features(other_row['dir'])
        sin_o, cos_o = get_angle_features(other_row['o'])
        
        # Feature "Coéquipier"
        is_teammate = 1.0 if other_row['player_side'] == side_target else 0.0
        
        k_feats = [
            (other_row['x'] - current_pos['x']) / MAX_DIST, # Relatif à moi
            (other_row['y'] - current_pos['y']) / MAX_DIST,
            other_row['s'] / MAX_SPEED,
            other_row['a'] / MAX_ACCEL,
            sin_d, cos_d,
            sin_o, cos_o,
            is_teammate # Feature 9
        ]
        keys_list.append(k_feats)
    
    # Padding à 22 (taille max contexte)
    while len(keys_list) < 22:
        keys_list.append([0.0] * 9) # 9 zéros
        
    keys = torch.tensor(keys_list[:22], dtype=torch.float32).unsqueeze(0) # (1, 22, 9)
    
    # Position de départ pour l'affichage (Absolue)
    start_pos = (current_pos['x'], current_pos['y'])
    
    return query, keys, start_pos

# --- 2. FONCTION D'ANIMATION FINALE ---
def animate_prediction_v4(model, df_in, df_out, game_id, play_id):
    
    print(f"🔍 Visualisation V4 | Game {game_id} Play {play_id}")
    play_in = df_in[(df_in['game_id'] == game_id) & (df_in['play_id'] == play_id)]
    play_out_full = df_out[(df_out['game_id'] == game_id) & (df_out['play_id'] == play_id)]
    
    if play_in.empty: return print("❌ Erreur : Input vide.")

    # Ballon
    ball_x, ball_y = play_in['ball_land_x'].iloc[0], play_in['ball_land_y'].iloc[0]
    players_to_predict = play_out_full['nfl_id'].unique()
    
    # Couleurs
    player_colors = {}
    for nid in play_in['nfl_id'].unique():
        info = play_in[play_in['nfl_id'] == nid].iloc[0]
        side = info['player_side']
        if nid in players_to_predict:
            player_colors[nid] = '#FFD700' if side == 'Offense' else '#FF00FF' # Gold / Magenta
        else:
            player_colors[nid] = '#1f77b4' if side == 'Offense' else '#d62728' # Bleu / Rouge

    # --- PRÉDICTIONS ---
    ai_trajectories = {} 
    real_trajectories = {} 
    model.eval()
    
    for nid in players_to_predict:
        # Réalité
        real_track = play_out_full[play_out_full['nfl_id'] == nid].sort_values('frame_id')
        real_trajectories[nid] = real_track
        
        # IA (Appel de la nouvelle fonction V4)
        query, keys, start_pos = prepare_input_v4(df_in, game_id, play_id, nid)
        
        if query is not None:
            with torch.no_grad():
                pred, _ = model(query, keys)
            
            # Dénormalisation : Ici on suppose que le label était un Offset brut (Yards)
            # Si vous aviez divisé le label par 50, faites : pred[0,0].item() * 50.0
            dx = pred[0, 0].item() 
            dy = pred[0, 1].item()
            
            pred_x = start_pos[0] + dx
            pred_y = start_pos[1] + dy
            
            steps = len(real_track)
            if steps > 0:
                ai_xs = np.linspace(start_pos[0], pred_x, steps)
                ai_ys = np.linspace(start_pos[1], pred_y, steps)
                ai_trajectories[nid] = (ai_xs, ai_ys)

    # --- GRAPHIQUE ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('#004d00')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    for x in range(10, 111, 10): ax.axvline(x, color='white', alpha=0.15)
    
    # Ballon
    ball_marker = ax.scatter([ball_x], [ball_y], marker='X', s=200, c='white', edgecolors='black', zorder=20)
    
    # Légende
    legend_elements = [
        Line2D([0], [0], marker='X', color='w', markeredgecolor='k', label='Ballon'),
        Line2D([0], [0], color='#FFD700', lw=4, alpha=0.5, label='Vrai (Att)'),
        Line2D([0], [0], color='#FFD700', lw=2, linestyle='--', label='IA (Att)'),
        Line2D([0], [0], color='#FF00FF', lw=4, alpha=0.5, label='Vrai (Def)'),
        Line2D([0], [0], color='#FF00FF', lw=2, linestyle='--', label='IA (Def)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='black', labelcolor='white')

    # Objets
    scats = {}
    for nid in play_in['nfl_id'].unique():
        c = player_colors.get(nid, 'white')
        s = 120 if nid in players_to_predict else 40
        scats[nid] = ax.scatter([], [], s=s, c=c, edgecolors='white', zorder=10)
        
    lines_real = {nid: ax.plot([], [], c=player_colors[nid], alpha=0.4, lw=6, zorder=8)[0] for nid in players_to_predict}
    lines_ai = {nid: ax.plot([], [], c=player_colors[nid], ls='--', lw=2, marker='.', markersize=4, zorder=9)[0] for nid in players_to_predict}

    # Animation
    max_in_frame = int(play_in['frame_id'].max()) if not play_in.empty else 0
    max_out_steps = max([len(t) for t in real_trajectories.values()]) if real_trajectories else 10
    total_frames = int(max_in_frame + max_out_steps)

    def update(frame_idx):
        frame_num = frame_idx + 1
        title_text = f"Frame {frame_num} | INPUT" if frame_num <= max_in_frame else f"Step {frame_num - max_in_frame} | PRED vs REAL"
        ax.set_title(title_text, color='white', backgroundcolor='black')
        
        artists = [ball_marker]
        
        if frame_num <= max_in_frame:
            # Input
            current_data = play_in[play_in['frame_id'] == frame_num]
            for _, row in current_data.iterrows():
                nid = row['nfl_id']
                if nid in scats:
                    scats[nid].set_offsets([[row['x'], row['y']]])
                    artists.append(scats[nid])
        else:
            # Output
            step = frame_num - max_in_frame - 1
            for nid in players_to_predict:
                # Real
                if nid in real_trajectories and step < len(real_trajectories[nid]):
                    path = real_trajectories[nid]
                    lines_real[nid].set_data(path.iloc[:step+1]['x'], path.iloc[:step+1]['y'])
                    artists.append(lines_real[nid])
                    # Update point
                    pos = path.iloc[step]
                    scats[nid].set_offsets([[pos['x'], pos['y']]])
                    artists.append(scats[nid])
                # IA
                if nid in ai_trajectories and step < len(ai_trajectories[nid][0]):
                    xs, ys = ai_trajectories[nid]
                    lines_ai[nid].set_data(xs[:step+1], ys[:step+1])
                    artists.append(lines_ai[nid])
                    
        return artists

    output_folder = "../results"
    filename = f"pred_game_existing_{game_id}_play_{play_id}.html"
    filepath = os.path.join(output_folder, filename)

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=100, blit=True)
    with open(filepath, "w") as f:
        f.write(ani.to_jshtml())

    plt.close()
    return HTML(ani.to_jshtml())

# --- CONSTANTES ---
HISTORY_SIZE = 10
MAX_SPEED = 13.0
MAX_ACCEL = 10.0
MAX_DIST = 50.0

def get_angle_features(degree_val):
    rad = np.radians(degree_val)
    return np.sin(rad), np.cos(rad)

# --- 1. PRÉPARATION INPUT TEST (Sans Output) ---
def prepare_test_input(df, game_id, play_id, target_nfl_id):
    play_data = df[(df['game_id'] == game_id) & (df['play_id'] == play_id)].sort_values('frame_id')
    player_track = play_data[play_data['nfl_id'] == target_nfl_id]
    
    if len(player_track) < HISTORY_SIZE: return None, None, None, None
        
    # Input Sequence
    history_seq = player_track.iloc[-HISTORY_SIZE:]
    current_pos = history_seq.iloc[-1]
    
    # 1. Query (90 features)
    rec_features = []
    ball_x, ball_y = current_pos['ball_land_x'], current_pos['ball_land_y']
    is_offense = 1.0 if current_pos['player_side'] == 'Offense' else 0.0
    
    for _, row in history_seq.iterrows():
        sin_d, cos_d = get_angle_features(row['dir'])
        sin_o, cos_o = get_angle_features(row['o'])
        feats = [
            row['s'] / MAX_SPEED, row['a'] / MAX_ACCEL,
            sin_d, cos_d, sin_o, cos_o,
            (ball_x - row['x']) / MAX_DIST, (ball_y - row['y']) / MAX_DIST,
            is_offense
        ]
        rec_features.extend(feats)
    query = torch.tensor(rec_features, dtype=torch.float32).unsqueeze(0)

    # 2. Keys (Contexte)
    current_frame_id = current_pos['frame_id']
    others = play_data[(play_data['frame_id'] == current_frame_id) & (play_data['nfl_id'] != target_nfl_id)]
    keys_list = []
    side_target = current_pos['player_side']
    
    for _, other_row in others.iterrows():
        sin_d, cos_d = get_angle_features(other_row['dir'])
        sin_o, cos_o = get_angle_features(other_row['o'])
        is_teammate = 1.0 if other_row['player_side'] == side_target else 0.0
        k_feats = [
            (other_row['x'] - current_pos['x']) / MAX_DIST, (other_row['y'] - current_pos['y']) / MAX_DIST,
            other_row['s'] / MAX_SPEED, other_row['a'] / MAX_ACCEL,
            sin_d, cos_d, sin_o, cos_o, is_teammate
        ]
        keys_list.append(k_feats)
    
    while len(keys_list) < 22: keys_list.append([0.0] * 9)
    keys = torch.tensor(keys_list[:22], dtype=torch.float32).unsqueeze(0)
    
    start_pos = (current_pos['x'], current_pos['y'])
    # On récupère le nombre de frames à prédire (spécifique au test set)
    num_frames_out = int(current_pos['num_frames_output'])
    
    return query, keys, start_pos, num_frames_out

# --- 2. ANIMATION TEST ---
def animate_test_prediction(model, test_input_df, test_df, game_id, play_id):
    print(f"🎬 Visualisation TEST | Game {game_id} Play {play_id}")
    
    # Filtrer les inputs
    play_in = test_input_df[(test_input_df['game_id'] == game_id) & (test_input_df['play_id'] == play_id)]
    if play_in.empty: return print("❌ Input vide.")

    # Qui doit-on prédire ? (Depuis le fichier test.csv ou player_to_predict=True)
    # Option 1 : Utiliser test.csv si dispo
    targets_in_test_file = test_df[(test_df['game_id'] == game_id) & (test_df['play_id'] == play_id)]['nfl_id'].unique()
    # Option 2 : Fallback sur input
    targets_in_input = play_in[play_in['player_to_predict'] == True]['nfl_id'].unique()
    
    players_to_predict = targets_in_test_file if len(targets_in_test_file) > 0 else targets_in_input
    print(f"   -> {len(players_to_predict)} joueurs à prédire.")

    # Ballon
    ball_x, ball_y = play_in['ball_land_x'].iloc[0], play_in['ball_land_y'].iloc[0]

    # Couleurs
    player_colors = {}
    for nid in play_in['nfl_id'].unique():
        info = play_in[play_in['nfl_id'] == nid].iloc[0]
        side = info['player_side']
        if nid in players_to_predict:
            player_colors[nid] = '#FFD700' if side == 'Offense' else '#FF00FF' # Gold / Magenta
        else:
            player_colors[nid] = '#1f77b4' if side == 'Offense' else '#d62728' # Bleu / Rouge

    # --- PRÉDICTIONS ---
    ai_trajectories = {}
    model.eval()
    max_steps_pred = 0

    for nid in players_to_predict:
        query, keys, start_pos, n_frames = prepare_test_input(test_input_df, game_id, play_id, nid)
        
        if query is not None:
            with torch.no_grad():
                pred, _ = model(query, keys)
            
            # Dénormalisation (Offset brut prédit)
            # Attention : Le modèle a appris à prédire la position FINALE relative.
            # On va interpoler linéairement jusqu'à cette position.
            dx = pred[0, 0].item()
            dy = pred[0, 1].item()
            
            pred_x = start_pos[0] + dx
            pred_y = start_pos[1] + dy
            
            steps = n_frames # Le nombre de frames demandé par le test set
            max_steps_pred = max(max_steps_pred, steps)
            
            ai_xs = np.linspace(start_pos[0], pred_x, steps)
            ai_ys = np.linspace(start_pos[1], pred_y, steps)
            ai_trajectories[nid] = (ai_xs, ai_ys)

    # --- GRAPHIQUE ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('#004d00')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    for x in range(10, 111, 10): ax.axvline(x, color='white', alpha=0.15)
    
    # Ballon
    ball_marker = ax.scatter([ball_x], [ball_y], marker='X', s=200, c='white', edgecolors='black', zorder=20)
    
    # Légende
    legend_elements = [
        Line2D([0], [0], marker='X', color='w', markeredgecolor='k', label='Ballon (Cible)'),
        Line2D([0], [0], color='#FFD700', lw=2, linestyle='--', marker='.', label='IA (Attaque)'),
        Line2D([0], [0], color='#FF00FF', lw=2, linestyle='--', marker='.', label='IA (Défense)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='black', labelcolor='white')

    # Objets
    scats = {}
    for nid in play_in['nfl_id'].unique():
        c = player_colors.get(nid, 'white')
        s = 120 if nid in players_to_predict else 40
        scats[nid] = ax.scatter([], [], s=s, c=c, edgecolors='white', zorder=10)
        
    lines_ai = {nid: ax.plot([], [], c=player_colors[nid], ls='--', lw=2, marker='.', markersize=4, zorder=9)[0] for nid in players_to_predict}

    # Animation
    max_in_frame = int(play_in['frame_id'].max())
    total_frames = int(max_in_frame + max_steps_pred)

    def update(frame_idx):
        frame_num = frame_idx + 1
        title_text = f"TEST SET | Frame {frame_num}"
        ax.set_title(title_text, color='white', backgroundcolor='black')
        artists = [ball_marker]
        
        if frame_num <= max_in_frame:
            # Phase Input
            current_data = play_in[play_in['frame_id'] == frame_num]
            for _, row in current_data.iterrows():
                nid = row['nfl_id']
                if nid in scats:
                    scats[nid].set_offsets([[row['x'], row['y']]])
                    artists.append(scats[nid])
        else:
            # Phase Prediction
            step = frame_num - max_in_frame - 1
            for nid in players_to_predict:
                if nid in ai_trajectories:
                    xs, ys = ai_trajectories[nid]
                    if step < len(xs):
                        # Tracer la ligne progressive
                        lines_ai[nid].set_data(xs[:step+1], ys[:step+1])
                        artists.append(lines_ai[nid])
                        # Bouger le point au bout de la ligne
                        scats[nid].set_offsets([[xs[step], ys[step]]])
                        artists.append(scats[nid])
                    
        return artists

    output_folder = "../results"
    filename = f"pred_game_{game_id}_play_{play_id}.html"
    filepath = os.path.join(output_folder, filename)

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=100, blit=True)
    with open(filepath, "w") as f:
        f.write(ani.to_jshtml())

    plt.close()
    return HTML(ani.to_jshtml())

