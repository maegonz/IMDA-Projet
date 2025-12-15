import pandas as pd
import torch
from typing import List
from data.processing import get_angle_features, HISTORY_SIZE, MAX_SPEED, MAX_ACCEL, MAX_DIST


def _pad_and_tensorize_keys(keys_list: List[List[float]], max_len: int = 22, feature_dim: int = 9):
    """
    Pad a variable-length list of per-player context features to a fixed
    length and convert it into a model-ready tensor.

    This ensures the keys tensor always has shape (1, max_len, feature_dim)
    by zero-padding when fewer than max_len players are present and
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
        Padded/truncated keys tensor of shape (1, max_len, feature_dim).
    """
    while len(keys_list) < max_len:
        keys_list.append([0.0] * feature_dim)

    return torch.tensor(keys_list[:max_len], dtype=torch.float32).unsqueeze(0)

def _player_historic(
        df: pd.DataFrame,
        game_id: int,
        play_id: int,
        nfl_id: int):
    """
    Retrieve the full play slice and the target player's recent history.

    Filters the play rows by game_id and play_id, then returns
    the last HISTORY_SIZE frames for nfl_id if available.

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
        (play_data, player_history) when sufficient history exists;
        otherwise None.
    """
    # Filter full play frames ordered by time
    play_data = df[(df['game_id'] == game_id) & (df['play_id'] == play_id)].sort_values('frame_id')
    # Player track
    player_track = play_data[play_data['nfl_id'] == nfl_id]
    assert not player_track.empty, f"============ ERROR ============ \n No data for player {nfl_id} in game {game_id}, play {play_id}."

    if len(player_track) < HISTORY_SIZE:
        print(f"Less thant {HISTORY_SIZE} history for player {nfl_id} in game {game_id}, play {play_id}.")
        print(f"Available frames: {len(player_track)}")
        return None, None

    return play_data, player_track.iloc[-HISTORY_SIZE:]


def _prepare_core_input(df, game_id, play_id, target_id, num_frames_out=False):
    """
    Build model inputs for a target player from tracking data.

    Constructs the receiver query by flattening the last HISTORY_SIZE
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
    target_id : int
        Player (target) identifier.
    num_frames_out : bool, default False
        When True, also return num_frames_out read from the current row.

    Returns
    -------
    tuple
        If num_frames_out is False: (query, keys, start_pos).
        If True: (query, keys, start_pos, num_frames_out).
        Returns (None, None, None) or (None, None, None, None) if
        insufficient history for the target player.
    """
    # Filter play data
    play_data, history_seq = _player_historic(df, game_id, play_id, target_id)
    if history_seq is None:
        return (None, None, None, None) if num_frames_out else (None, None, None)
    
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
                       (play_data['nfl_id'] != target_id)]

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


def prepare_input(
        df: pd.DataFrame,
        game_id: int,
        play_id: int,
        target_id: int):
    """
    Convenience wrapper to prepare training/inference inputs for a player.

    Builds the query and keys tensors for target_id within the given
    game_id and play_id. Intended for cases where output length is
    determined by ground-truth and not required in inputs.

    Returns
    -------
    tuple
        (query, keys, start_pos) where query has shape (1, 90),
        keys has shape (1, 22, 9), and start_pos is the absolute
        (x, y) position at the last history frame. Returns
        (None, None, None) when insufficient history.
    """
    return _prepare_core_input(df, game_id, play_id, target_id)


def prepare_test_input(
        df: pd.DataFrame,
        game_id: int,
        play_id: int,
        target_id: int):
    """
    Prepare inputs for test-time prediction including output length.

    Similar to prepare_input but also returns num_frames_out derived
    from the final history row, representing how many future frames to
    predict for the target player.

    Returns
    -------
    tuple
        (query, keys, start_pos, num_frames_out) or (None, None, None, None)
        when insufficient history. query shape: (1, 90); keys shape:
        (1, 22, 9); start_pos: absolute (x, y); num_frames_out:
        integer count of future frames to generate.
    """    
    return _prepare_core_input(df, game_id, play_id, target_id, num_frames_out=True)