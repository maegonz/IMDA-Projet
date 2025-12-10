import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

def create_football_field(ax):
    # Fond vert
    ax.set_facecolor('darkgreen')
    ax.add_patch(patches.Rectangle((0, 0), 120, 53.3, facecolor='darkgreen', zorder=0))

    # Lignes (tous les 10 yards)
    for x in range(10, 111, 10):
        ax.axvline(x, color='white', linestyle='-', alpha=0.3)

    # Endzones (0-10 et 110-120)
    ax.add_patch(patches.Rectangle((0, 0), 10, 53.3, facecolor='blue', alpha=0.2))
    ax.add_patch(patches.Rectangle((110, 0), 10, 53.3, facecolor='red', alpha=0.2))

    ax.set_xlim(0, 120)
    ax.set_ylim(0, 53.3)
    ax.set_aspect('equal')
    ax.axis('off')


def update(frame_num, player_plots):
    # frame_num commence à 0, mais vos frame_id commencent souvent à 1
    current_frame_id = frame_num + 1

    for _, (scat, text, group) in player_plots.items():
        # Trouver la ligne correspondant à cette frame
        data = group[group['frame_id'] == current_frame_id]

        if not data.empty:
            x = data['x'].values[0]
            y = data['y'].values[0]
            name = data['player_name'].values[0]

            scat.set_offsets([[x, y]])
            text.set_position((x, y + 1.5)) # Texte un peu au dessus
            text.set_text(name)
        else:
            # Si pas de donnée pour ce joueur à cette frame (ex: sortie du champ)
            scat.set_offsets([[-10, -10]]) # Hors champ
            text.set_text('')

    return [p[0] for p in player_plots.values()] + [p[1] for p in player_plots.values()]

