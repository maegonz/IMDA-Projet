import numpy as np
import torch
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
from .processing import prepare_input, prepare_test_input


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

    Uses df_in (input tracking) to render the input phase, then overlays
    the model's predicted path against the true path from df_out for
    the players that have labeled outputs in that play.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model that maps (query, keys) to predicted offsets.
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

    print(f"🔍 Game {game_id} Play {play_id}")

    if df_in.empty:
        return print("❌ input empty")

    predict_data = df_out[(df_out['game_id'] == game_id) & (df_out['play_id'] == play_id)].sort_values('frame_id')
    players_to_predict = predict_data['nfl_id'].unique()

    # COLORS
    player_colors = {}
    for nid in df_in['nfl_id'].unique():
        side = df_in.loc[df_in['nfl_id']==nid,'player_side'].iloc[0]
        if nid in players_to_predict:
            player_colors[nid] = '#FFD700' if side=='Offense' else '#FF00FF'
        else:
            player_colors[nid] = '#1f77b4' if side=='Offense' else '#d62728'

    ball_x, ball_y = df_in['ball_land_x'].iloc[0], df_in['ball_land_y'].iloc[0]

    # REAL trajectories
    trajectories_real = {
        nid: df_out[df_out['nfl_id']==nid].sort_values('frame_id')
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
        play_in=df_in,
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

    Renders the input phase from test_input_df and then shows the
    predicted trajectories for target players determined from test_df
    (if available) or the input flag player_to_predict.

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