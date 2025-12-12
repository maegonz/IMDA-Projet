from typing import Dict, List, Optional, Tuple

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
from data.processing import get_angle_features, HISTORY_SIZE, MAX_SPEED, MAX_ACCEL, MAX_DIST


def _pad_and_tensorize_keys(keys_list: List[List[float]], max_len: int = 22, feature_dim: int = 9):
    """
    Pad context features to a fixed length and return as tensor.

    Parameters
    ----------
    keys_list : list of list of float
        Per-player context features.
    max_len : int, default 22
        Target number of context players (padded with zeros when fewer).
    feature_dim : int, default 9
        Number of features per player.

    Returns
    -------
    torch.Tensor
        Tensor of shape (1, max_len, feature_dim).
    """
    while len(keys_list) < max_len:
        keys_list.append([0.0] * feature_dim)

    return torch.tensor(keys_list[:max_len], dtype=torch.float32).unsqueeze(0)


def _prepare_core_input(df, game_id, play_id, target_nfl_id, num_frames_out=False):
    """
    Shared core function for preparing model inputs.
    Used by both training and testing wrappers.
    """
    # Filter play data
    play_data, history_seq = _player_historic(df, game_id, play_id, target_nfl_id)
    if history_seq is None:
        return None, None, None, None if num_frames_out else (None, None, None)
    
    # Full play data for context lookups
    current_pos = history_seq.iloc[-1]  # Last frame (T)

    # --- A. Build receiver query (flattened history) ---
    # Expected size: 90 (HISTORY_SIZE * 9 features)
    rec_features = []
    ball_x, ball_y = current_pos['ball_land_x'], current_pos['ball_land_y']
    is_offense = 1.0 if current_pos['player_side'] == 'Offense' else 0.0

    for _, row in history_seq.iterrows():
        sin_d, cos_d = get_angle_features(row['dir'])
        sin_o, cos_o = get_angle_features(row['o'])

        rec_features.extend([
            row['s'] / MAX_SPEED,
            row['a'] / MAX_ACCEL,
            sin_d, cos_d,
            sin_o, cos_o,
            (ball_x - row['x']) / MAX_DIST,
            (ball_y - row['y']) / MAX_DIST,
            is_offense  # Feature 9: offense flag for target
        ])

    query = torch.tensor(rec_features, dtype=torch.float32).unsqueeze(0)

    # --- B. Build context keys (teammates + opponents at last frame) ---
    current_frame_id = current_pos['frame_id']
    others = play_data[(play_data['frame_id'] == current_frame_id) & 
                       (play_data['nfl_id'] != target_nfl_id)]

    keys_list = []  # List[List[float]] for context players
    side_target = current_pos['player_side']

    for _, other in others.iterrows():
        sin_d, cos_d = get_angle_features(other['dir'])
        sin_o, cos_o = get_angle_features(other['o'])

        is_teammate = 1.0 if other['player_side'] == side_target else 0.0

        keys_list.append([
            (other['x'] - current_pos['x']) / MAX_DIST,
            (other['y'] - current_pos['y']) / MAX_DIST,
            other['s'] / MAX_SPEED,
            other['a'] / MAX_ACCEL,
            sin_d, cos_d,
            sin_o, cos_o,
            is_teammate
        ])

    # Pad to 22 context players (max seen in play)
    keys = _pad_and_tensorize_keys(keys_list, max_len=22, feature_dim=9)  # (1, 22, 9)
    
    # Absolute starting position for visualization
    start_pos = (current_pos['x'], current_pos['y'])

    # Optional: number of future frames to predict
    if num_frames_out:
        num_frames_out = int(current_pos['num_frames_output'])
        return query, keys, start_pos, num_frames_out

    return query, keys, start_pos


def _player_historic(
        df: pd.DataFrame,
        game_id: int,
        play_id: int,
        nfl_id: int):
    """
    Fetch the last HISTORY_SIZE frames for a player in a given play.

    Parameters
    ----------
    df : pandas.DataFrame
        Full tracking dataframe containing the play.
    game_id : int
        Game identifier to filter rows.
    play_id : int
        Play identifier to filter rows.
    nfl_id : int
        Player identifier whose history is requested.

    Returns
    -------
    pandas.DataFrame or None
        The last ``HISTORY_SIZE`` frames for the player, or ``None`` when not enough history.
    """
    # Filter full play frames ordered by time
    play_data = df[(df['game_id'] == game_id) & (df['play_id'] == play_id)].sort_values('frame_id')
    # Player track
    player_track = play_data[play_data['nfl_id'] == nfl_id]

    if len(player_track) < HISTORY_SIZE:
        return None

    return play_data, player_track.iloc[-HISTORY_SIZE:]

def prepare_input(
        df: pd.DataFrame,
        game_id: int,
        play_id: int,
        target_nfl_id: int):
    """
    Build model inputs (history + context) for a target player.

    Parameters
    ----------
    df : pandas.DataFrame
        Tracking dataframe with input play data.
    game_id : int
        Game identifier to filter rows.
    play_id : int
        Play identifier to filter rows.
    target_nfl_id : int
        Player identifier to prepare inputs for.

    Returns
    -------
    tuple
        (query, keys, start_pos) where ``query`` is shape (1, 90), ``keys`` is
        shape (1, 22, 9), and ``start_pos`` is the absolute (x, y) starting point.
        Returns (None, None, None) when insufficient history.
    """
    return _prepare_core_input(df, game_id, play_id, target_nfl_id)


def prepare_test_input(
        df: pd.DataFrame,
        game_id: int,
        play_id: int,
        target_nfl_id: int):
    """
    Prepare model inputs for test scenarios without known outputs.

    Parameters
    ----------
    df : pandas.DataFrame
        Tracking dataframe containing the play inputs.
    game_id : int
        Game identifier to filter rows.
    play_id : int
        Play identifier to filter rows.
    target_nfl_id : int
        Player identifier whose future path is predicted.

    Returns
    -------
    tuple
        (query, keys, start_pos, num_frames_out) or (None, None, None, None) when
        insufficient history. ``query`` shape: (1, 90); ``keys`` shape: (1, 22, 9);
        ``start_pos``: absolute (x, y); ``num_frames_out``: frames to predict.
    """    
    return _prepare_core_input(df, game_id, play_id, target_nfl_id, num_frames_out=True)


def animate_prediction(model, df_in, df_out, game_id, play_id):
    """Animate model predictions versus ground truth for a given play."""

    print(f"🔍 Visualisation | Game {game_id} Play {play_id}")
    play_in = df_in[(df_in['game_id'] == game_id) & (df_in['play_id'] == play_id)]
    play_out_full = df_out[(df_out['game_id'] == game_id) & (df_out['play_id'] == play_id)]
    
    if play_in.empty: return print("❌ Error: empty input.")

    # Ballon
    ball_x, ball_y = play_in['ball_land_x'].iloc[0], play_in['ball_land_y'].iloc[0]
    players_to_predict = play_out_full['nfl_id'].unique()
    
    # Colors per player depending on side and prediction target
    player_colors = {}
    for nid in play_in['nfl_id'].unique():
        info = play_in[play_in['nfl_id'] == nid].iloc[0]
        side = info['player_side']
        if nid in players_to_predict:
            player_colors[nid] = '#FFD700' if side == 'Offense' else '#FF00FF' # Gold / Magenta
        else:
            player_colors[nid] = '#1f77b4' if side == 'Offense' else '#d62728' # Bleu / Rouge

    # --- PREDICTIONS ---
    ai_trajectories = {} 
    real_trajectories = {} 
    model.eval()
    
    for nid in players_to_predict:
        # Ground truth trajectory
        real_track = play_out_full[play_out_full['nfl_id'] == nid].sort_values('frame_id')
        real_trajectories[nid] = real_track
        
        # Model prediction using inputs
        query, keys, start_pos = prepare_input(df_in, game_id, play_id, nid)
        
        if query is not None:
            with torch.no_grad():
                pred, _ = model(query, keys)
            
            # Denormalization: assume label was a raw yard offset (adjust if scaled)
            dx = pred[0, 0].item() 
            dy = pred[0, 1].item()
            
            pred_x = start_pos[0] + dx
            pred_y = start_pos[1] + dy
            
            steps = len(real_track)
            if steps > 0:
                ai_xs = np.linspace(start_pos[0], pred_x, steps)
                ai_ys = np.linspace(start_pos[1], pred_y, steps)
                ai_trajectories[nid] = (ai_xs, ai_ys)

    # --- FIGURE SETUP ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('#004d00')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    for x in range(10, 111, 10): ax.axvline(x, color='white', alpha=0.15)
    
    # Ball marker
    ball_marker = ax.scatter([ball_x], [ball_y], marker='X', s=200, c='white', edgecolors='black', zorder=20)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='X', color='w', markeredgecolor='k', label='Ballon'),
        Line2D([0], [0], color='#FFD700', lw=4, alpha=0.5, label='Vrai (Att)'),
        Line2D([0], [0], color='#FFD700', lw=2, linestyle='--', label='IA (Att)'),
        Line2D([0], [0], color='#FF00FF', lw=4, alpha=0.5, label='Vrai (Def)'),
        Line2D([0], [0], color='#FF00FF', lw=2, linestyle='--', label='IA (Def)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='black', labelcolor='white')

    # Scatter and line objects for animation
    scats = {}
    for nid in play_in['nfl_id'].unique():
        c = player_colors.get(nid, 'white')
        s = 120 if nid in players_to_predict else 40
        scats[nid] = ax.scatter([], [], s=s, c=c, edgecolors='white', zorder=10)
        
    lines_real = {nid: ax.plot([], [], c=player_colors[nid], alpha=0.4, lw=6, zorder=8)[0] for nid in players_to_predict}
    lines_ai = {nid: ax.plot([], [], c=player_colors[nid], ls='--', lw=2, marker='.', markersize=4, zorder=9)[0] for nid in players_to_predict}

    # Animation timing
    max_in_frame = int(play_in['frame_id'].max()) if not play_in.empty else 0
    max_out_steps = max([len(t) for t in real_trajectories.values()]) if real_trajectories else 10
    total_frames = int(max_in_frame + max_out_steps)

    def update(frame_idx):
        frame_num = frame_idx + 1
        title_text = f"Frame {frame_num} | INPUT" if frame_num <= max_in_frame else f"Step {frame_num - max_in_frame} | PRED vs REAL"
        ax.set_title(title_text, color='white', backgroundcolor='black')
        
        artists = [ball_marker]
        
        if frame_num <= max_in_frame:
            # Input phase
            current_data = play_in[play_in['frame_id'] == frame_num]
            for _, row in current_data.iterrows():
                nid = row['nfl_id']
                if nid in scats:
                    scats[nid].set_offsets([[row['x'], row['y']]])
                    artists.append(scats[nid])
        else:
            # Output phase (pred vs real)
            step = frame_num - max_in_frame - 1
            for nid in players_to_predict:
                # Ground truth path
                if nid in real_trajectories and step < len(real_trajectories[nid]):
                    path = real_trajectories[nid]
                    lines_real[nid].set_data(path.iloc[:step+1]['x'], path.iloc[:step+1]['y'])
                    artists.append(lines_real[nid])
                    # Update point
                    pos = path.iloc[step]
                    scats[nid].set_offsets([[pos['x'], pos['y']]])
                    artists.append(scats[nid])
                # Model path
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


# --- 2. ANIMATION TEST ---
def animate_test_prediction(model, test_input_df, test_df, game_id, play_id):
    """Animate predictions on test inputs where targets are unknown at runtime."""

    print(f"🎬 Visualisation TEST | Game {game_id} Play {play_id}")
    
    # Filter inputs for the requested play
    play_in = test_input_df[(test_input_df['game_id'] == game_id) & (test_input_df['play_id'] == play_id)]
    if play_in.empty:
        return print("❌ Empty input.")

    # Determine players to predict (prefer test.csv, fallback to player_to_predict flag)
    # Option 1: use test.csv if available
    targets_in_test_file = test_df[(test_df['game_id'] == game_id) & (test_df['play_id'] == play_id)]['nfl_id'].unique()
    # Option 2: fallback to input flag
    targets_in_input = play_in[play_in['player_to_predict'] == True]['nfl_id'].unique()
    
    players_to_predict = targets_in_test_file if len(targets_in_test_file) > 0 else targets_in_input
    print(f"   -> {len(players_to_predict)} players to predict.")

    # Ball location
    ball_x, ball_y = play_in['ball_land_x'].iloc[0], play_in['ball_land_y'].iloc[0]

    # Colors per player based on side and prediction target
    player_colors = {}
    for nid in play_in['nfl_id'].unique():
        info = play_in[play_in['nfl_id'] == nid].iloc[0]
        side = info['player_side']
        if nid in players_to_predict:
            player_colors[nid] = '#FFD700' if side == 'Offense' else '#FF00FF'  # Gold / Magenta
        else:
            player_colors[nid] = '#1f77b4' if side == 'Offense' else '#d62728'  # Blue / Red

    # --- PREDICTIONS ---
    ai_trajectories = {}
    model.eval()
    max_steps_pred = 0

    for nid in players_to_predict:
        query, keys, start_pos, n_frames = prepare_test_input(test_input_df, game_id, play_id, nid)
        
        if query is not None:
            with torch.no_grad():
                pred, _ = model(query, keys)
            
            # Denormalize: model predicts final relative offset; we interpolate linearly to that point
            dx = pred[0, 0].item()
            dy = pred[0, 1].item()
            
            pred_x = start_pos[0] + dx
            pred_y = start_pos[1] + dy
            
            steps = n_frames  # Frames requested by the test set
            max_steps_pred = max(max_steps_pred, steps)
            
            ai_xs = np.linspace(start_pos[0], pred_x, steps)
            ai_ys = np.linspace(start_pos[1], pred_y, steps)
            ai_trajectories[nid] = (ai_xs, ai_ys)

    # --- FIGURE SETUP ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('#004d00')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    for x in range(10, 111, 10): ax.axvline(x, color='white', alpha=0.15)
    
    # Ballon
    ball_marker = ax.scatter([ball_x], [ball_y], marker='X', s=200, c='white', edgecolors='black', zorder=20)
    
    # Legend
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

    # Animation timing
    max_in_frame = int(play_in['frame_id'].max())
    total_frames = int(max_in_frame + max_steps_pred)

    def update(frame_idx):
        frame_num = frame_idx + 1
        title_text = f"TEST SET | Frame {frame_num}"
        ax.set_title(title_text, color='white', backgroundcolor='black')
        artists = [ball_marker]
        
        if frame_num <= max_in_frame:
            # Input phase
            current_data = play_in[play_in['frame_id'] == frame_num]
            for _, row in current_data.iterrows():
                nid = row['nfl_id']
                if nid in scats:
                    scats[nid].set_offsets([[row['x'], row['y']]])
                    artists.append(scats[nid])
        else:
            # Prediction phase
            step = frame_num - max_in_frame - 1
            for nid in players_to_predict:
                if nid in ai_trajectories:
                    xs, ys = ai_trajectories[nid]
                    if step < len(xs):
                        # Draw progressive predicted path
                        lines_ai[nid].set_data(xs[:step+1], ys[:step+1])
                        artists.append(lines_ai[nid])
                        # Move marker to path tip
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