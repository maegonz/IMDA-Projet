from typing import List

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
    Pad a variable-length list of per-player context features to a fixed
    length and convert it into a model-ready tensor.

    This ensures the keys tensor always has shape ``(1, max_len, feature_dim)``
    by zero-padding when fewer than ``max_len`` players are present and
    truncating when more.

    Parameters
    ----------
    keys_list : list[list[float]]
        Context features for each non-target player at the reference frame.
    max_len : int, default 22
        Fixed number of context players to represent.
    feature_dim : int, default 9
        Number of features per player key.

    Returns
    -------
    torch.Tensor
        Padded/truncated keys tensor of shape ``(1, max_len, feature_dim)``.
    """
    while len(keys_list) < max_len:
        keys_list.append([0.0] * feature_dim)

    return torch.tensor(keys_list[:max_len], dtype=torch.float32).unsqueeze(0)


def _prepare_core_input(df, game_id, play_id, target_nfl_id, num_frames_out=False):
    """
    Build model inputs for a target player from tracking data.

    Constructs the receiver query by flattening the last ``HISTORY_SIZE``
    frames and the context keys from all other players at the final history
    frame. Optionally returns the number of output frames when available
    (for test-time inputs).

    Parameters
    ----------
    df : pandas.DataFrame
        Tracking dataframe containing the play.
    game_id : int
        Game identifier.
    play_id : int
        Play identifier.
    target_nfl_id : int
        Player (target) identifier.
    num_frames_out : bool, default False
        When True, also return ``num_frames_out`` read from the current row.

    Returns
    -------
    tuple
        If ``num_frames_out`` is False: ``(query, keys, start_pos)``.
        If True: ``(query, keys, start_pos, num_frames_out)``.
        Returns ``(None, None, None)`` or ``(None, None, None, None)`` if
        insufficient history for the target player.
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
    Retrieve the full play slice and the target player's recent history.

    Filters the play rows by ``game_id`` and ``play_id``, then returns
    the last ``HISTORY_SIZE`` frames for ``nfl_id`` if available.

    Parameters
    ----------
    df : pandas.DataFrame
        Full tracking dataframe.
    game_id : int
        Game identifier.
    play_id : int
        Play identifier.
    nfl_id : int
        Target player identifier.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame] or None
        ``(play_data, player_history)`` when sufficient history exists;
        otherwise ``None``.
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
    Convenience wrapper to prepare training/inference inputs for a player.

    Builds the query and keys tensors for ``target_nfl_id`` within the given
    ``game_id`` and ``play_id``. Intended for cases where output length is
    determined by ground-truth and not required in inputs.

    Returns
    -------
    tuple
        ``(query, keys, start_pos)`` where ``query`` has shape ``(1, 90)``,
        ``keys`` has shape ``(1, 22, 9)``, and ``start_pos`` is the absolute
        ``(x, y)`` position at the last history frame. Returns
        ``(None, None, None)`` when insufficient history.
    """
    return _prepare_core_input(df, game_id, play_id, target_nfl_id)


def prepare_test_input(
        df: pd.DataFrame,
        game_id: int,
        play_id: int,
        target_nfl_id: int):
    """
    Prepare inputs for test-time prediction including output length.

    Similar to ``prepare_input`` but also returns ``num_frames_out`` derived
    from the final history row, representing how many future frames to
    predict for the target player.

    Returns
    -------
    tuple
        ``(query, keys, start_pos, num_frames_out)`` or ``(None, None, None, None)``
        when insufficient history. ``query`` shape: ``(1, 90)``; ``keys`` shape:
        ``(1, 22, 9)``; ``start_pos``: absolute ``(x, y)``; ``num_frames_out``:
        integer count of future frames to generate.
    """    
    return _prepare_core_input(df, game_id, play_id, target_nfl_id, num_frames_out=True)




def _animate_core(
        play_in,
        players_to_predict,
        player_colors,
        ball_x,
        ball_y,
        trajectories_pred,
        trajectories_real=None,
        title_prefix="",
        output_name="animation.html"):
    """
    Core animation engine shared by training/test visualization.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor('#004d00')
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)

    # Field grid lines
    for x in range(10, 111, 10):
        ax.axvline(x, color='white', alpha=0.15)

    # Ball marker
    ball_marker = ax.scatter(
        [ball_x], [ball_y], marker='X', s=200,
        c='white', edgecolors='black', zorder=20
    )

    # Prepare scatter objects (all players)
    scats = {}
    for nid in play_in['nfl_id'].unique():
        c = player_colors[nid]
        s = 120 if nid in players_to_predict else 40
        scats[nid] = ax.scatter([], [], s=s, c=c, edgecolors='white', zorder=10)

    # Lines for predicted paths
    lines_pred = {
        nid: ax.plot([], [], c=player_colors[nid],
                     ls='--', lw=2, marker='.', markersize=4, zorder=9)[0]
        for nid in players_to_predict
    }

    # Lines for real trajectories (only if provided)
    lines_real = {}
    if trajectories_real is not None:
        lines_real = {
            nid: ax.plot([], [], c=player_colors[nid],
                         alpha=0.4, lw=6, zorder=8)[0]
            for nid in players_to_predict
        }

    # Animation frames count
    max_in_frame = int(play_in['frame_id'].max())
    max_out_steps = max(len(traj[0]) for traj in trajectories_pred.values()) \
                    if trajectories_pred else 0

    total_frames = max_in_frame + max_out_steps

    # ----------- UPDATE FUNCTION -----------
    def update(frame_idx):
        frame_num = frame_idx + 1
        artists = [ball_marker]

        if frame_num <= max_in_frame:
            # INPUT PHASE
            ax.set_title(f"{title_prefix} | Input Frame {frame_num}",
                         color='white', backgroundcolor='black')
            frame_data = play_in[play_in['frame_id'] == frame_num]

            for _, row in frame_data.iterrows():
                nid = row['nfl_id']
                scats[nid].set_offsets([[row['x'], row['y']]])
                artists.append(scats[nid])

        else:
            # OUTPUT PHASE
            step = frame_num - max_in_frame - 1
            ax.set_title(f"{title_prefix} | Step {step}",
                         color='white', backgroundcolor='black')

            for nid in players_to_predict:

                # REAL TRAJECTORY (training only)
                if trajectories_real is not None:
                    real_track = trajectories_real[nid]
                    if step < len(real_track):
                        lines_real[nid].set_data(
                            real_track.iloc[:step+1]['x'],
                            real_track.iloc[:step+1]['y']
                        )
                        artists.append(lines_real[nid])
                        # Real point
                        scats[nid].set_offsets(
                            [[real_track.iloc[step]['x'], real_track.iloc[step]['y']]]
                        )
                        artists.append(scats[nid])

                # PREDICTED TRAJECTORY
                if step < len(trajectories_pred[nid][0]):
                    xs, ys = trajectories_pred[nid]
                    lines_pred[nid].set_data(xs[:step+1], ys[:step+1])
                    artists.append(lines_pred[nid])

                    # Predicted point
                    scats[nid].set_offsets([[xs[step], ys[step]]])
                    artists.append(scats[nid])

        return artists

    # ------------- WRITE OUTPUT HTML -------------
    output_folder = "../results"
    os.makedirs(output_folder, exist_ok=True)

    filepath = os.path.join(output_folder, output_name)
    ani = animation.FuncAnimation(fig, update,
                                  frames=total_frames, interval=100, blit=True)
    with open(filepath, "w") as f:
        f.write(ani.to_jshtml())

    plt.close()
    return HTML(ani.to_jshtml())


def animate_prediction(model, df_in, df_out, game_id, play_id):
    """
    Animate predicted versus ground-truth trajectories for one play.

    Uses ``df_in`` (input tracking) to render the input phase, then overlays
    the model's predicted path against the true path from ``df_out`` for
    the players that have labeled outputs in that play.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model that maps ``(query, keys)`` to predicted offsets.
    df_in : pandas.DataFrame
        Input tracking dataframe with history and context.
    df_out : pandas.DataFrame
        Output dataframe containing ground-truth future positions.
    game_id : int
        Game identifier.
    play_id : int
        Play identifier.

    Returns
    -------
    IPython.display.HTML
        HTML object embedding the interactive animation.
    """

    print(f"🔍 TRAIN V4 | Game {game_id} Play {play_id}")

    play_in = df_in[(df_in['game_id']==game_id)&(df_in['play_id']==play_id)]
    play_out = df_out[(df_out['game_id']==game_id)&(df_out['play_id']==play_id)]

    if play_in.empty:
        return print("❌ input empty")

    players_to_predict = play_out['nfl_id'].unique()

    # COLORS
    player_colors = {}
    for nid in play_in['nfl_id'].unique():
        side = play_in.loc[play_in['nfl_id']==nid,'player_side'].iloc[0]
        if nid in players_to_predict:
            player_colors[nid] = '#FFD700' if side=='Offense' else '#FF00FF'
        else:
            player_colors[nid] = '#1f77b4' if side=='Offense' else '#d62728'

    ball_x, ball_y = play_in['ball_land_x'].iloc[0], play_in['ball_land_y'].iloc[0]

    # REAL trajectories
    trajectories_real = {
        nid: play_out[play_out['nfl_id']==nid].sort_values('frame_id')
        for nid in players_to_predict
    }

    # PRED trajectories
    trajectories_pred = {}
    model.eval()
    for nid in players_to_predict:
        query, keys, start_pos = prepare_input(df_in, game_id, play_id, nid)
        if query is None: continue

        with torch.no_grad():
            pred, _ = model(query, keys)
        dx, dy = pred[0].tolist()

        # Number of real steps
        steps = len(trajectories_real[nid])
        xs = np.linspace(start_pos[0], start_pos[0] + dx, steps)
        ys = np.linspace(start_pos[1], start_pos[1] + dy, steps)
        trajectories_pred[nid] = (xs, ys)

    return _animate_core(
        play_in=play_in,
        players_to_predict=players_to_predict,
        player_colors=player_colors,
        ball_x=ball_x,
        ball_y=ball_y,
        trajectories_pred=trajectories_pred,
        trajectories_real=trajectories_real,
        title_prefix="TRAIN",
        output_name=f"pred_train_{game_id}_{play_id}.html"
    )


# --- 2. ANIMATION TEST ---
def animate_test_prediction(model, test_input_df, test_df, game_id, play_id):
    """
    Animate model-only predictions for a test play without ground truth.

    Renders the input phase from ``test_input_df`` and then shows the
    predicted trajectories for target players determined from ``test_df``
    (if available) or the input flag ``player_to_predict``.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model for generating trajectory offsets.
    test_input_df : pandas.DataFrame
        Test inputs with history/context for the requested play.
    test_df : pandas.DataFrame
        Test metadata indicating target players and requested output length.
    game_id : int
        Game identifier.
    play_id : int
        Play identifier.

    Returns
    -------
    IPython.display.HTML
        HTML object embedding the interactive animation.
    """
    print(f"🎬 TEST | Game {game_id} Play {play_id}")

    # Filter play
    play_in = test_input_df[(test_input_df['game_id']==game_id)&
                            (test_input_df['play_id']==play_id)]
    if play_in.empty:
        return print("❌ input empty")

    # Determine players to predict
    targets_test = test_df[(test_df['game_id']==game_id)&
                           (test_df['play_id']==play_id)]['nfl_id'].unique()

    players_to_predict = targets_test

    # Colors
    player_colors = {}
    for nid in play_in['nfl_id'].unique():
        side = play_in.loc[play_in['nfl_id']==nid, 'player_side'].iloc[0]
        if nid in players_to_predict:
            player_colors[nid] = '#FFD700' if side=='Offense' else '#FF00FF'
        else:
            player_colors[nid] = '#1f77b4' if side=='Offense' else '#d62728'

    # BALL
    ball_x, ball_y = play_in['ball_land_x'].iloc[0], play_in['ball_land_y'].iloc[0]

    # ------------- PREDICT -------------
    trajectories_pred = {}
    model.eval()
    for nid in players_to_predict:
        query, keys, start_pos, n_frames = prepare_test_input(
            test_input_df, game_id, play_id, nid
        )
        if query is None:
            continue
        with torch.no_grad():
            pred, _ = model(query, keys)

        dx, dy = pred[0].tolist()
        end_x = start_pos[0] + dx
        end_y = start_pos[1] + dy

        xs = np.linspace(start_pos[0], end_x, n_frames)
        ys = np.linspace(start_pos[1], end_y, n_frames)
        trajectories_pred[nid] = (xs, ys)

    return _animate_core(
        play_in=play_in,
        players_to_predict=players_to_predict,
        player_colors=player_colors,
        ball_x=ball_x,
        ball_y=ball_y,
        trajectories_pred=trajectories_pred,
        trajectories_real=None,
        title_prefix="TEST",
        output_name=f"pred_test_{game_id}_{play_id}.html"
    )