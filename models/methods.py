import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch import nn, optim
from torch.utils.data import DataLoader
from data.processing import normalisation

def train_model(model: nn.Module,
                train_loader: DataLoader,
                criterion,
                optimizer: optim.Optimizer,
                num_epochs: int=20):
    
    # 4. Entraînement
    print("🚀 Entraînement avec Normalisation...")
    losses = []

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for q, k, y in train_loader:
            optimizer.zero_grad()
            preds, _ = model(q, k)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)

        # La loss sera maintenant très petite (ex: 0.05) car tout est divisé
        # Pour avoir l'erreur en Yards, on remultiplie par 10 (l'échelle du label)
        # MSE -> RMSE -> * 10
        rmse_yards = (avg_loss**0.5) * 10.0

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Erreur estimée: {rmse_yards:.2f} yards")

    plt.plot(losses)
    plt.title("Loss (Normalisée)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   criterion):
    
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for q, k, y in test_loader:
            preds, _ = model(q, k)
            loss = criterion(preds, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    rmse_yards = (avg_loss**0.5) * 10.0
    print(f"Évaluation | Loss: {avg_loss:.4f} | Erreur estimée: {rmse_yards:.2f} yards")


def analyze_play(model: nn.Module, idx: int):
    # Récupérer les données brutes (avant normalisation pour l'affichage)
    raw_data = torch.load('nfl_attention_dataset_CLEAN.pt')
    q_raw = raw_data['queries'][idx]
    k_raw = raw_data['keys'][idx]

    # Préparer les données pour le modèle (AVEC normalisation)
    q_norm = q_raw.clone()
    k_norm = k_raw.clone()

    q_norm, k_norm = normalisation(q_norm, k_norm)

    # Ajouter la dimension Batch (1, ...)
    q_input = q_norm.unsqueeze(0)
    k_input = k_norm.unsqueeze(0)

    # --- PRÉDICTION ---
    model.eval()
    with torch.no_grad():
        preds, attn_weights = model(q_input, k_input)

    # Poids d'attention (L'importance de chaque défenseur)
    weights = attn_weights[0, 0, :].numpy()

    # --- DESSIN ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Le Receveur (Toujours au centre 0,0 dans notre référentiel)
    ax.scatter(0, 0, c='blue', s=200, label='Receveur', zorder=5, edgecolors='white')

    # 2. Le Ballon (Cible)
    # Info stockée dans query [.., .., DistX, DistY]
    ball_x = q_raw[2].item()
    ball_y = q_raw[3].item()
    ax.scatter(ball_x, ball_y, c='gold', marker='x', s=150, linewidth=3, label='Ballon (Cible)')

    # 3. La Prédiction (Où l'IA pense que le receveur va aller)
    pred_dx = preds[0, 0].item() * 10.0
    pred_dy = preds[0, 1].item() * 10.0
    ax.arrow(0, 0, pred_dx, pred_dy, head_width=1, head_length=1, fc='cyan', ec='cyan', label='Trajectoire Prédite')

    # 4. Les Défenseurs (Colorés selon l'Attention)
    # Info stockée dans keys [RelX, RelY, ..]
    def_x = k_raw[:, 0].numpy()
    def_y = k_raw[:, 1].numpy()

    sc = ax.scatter(def_x, def_y, c=weights, cmap='Reds', s=100 + (weights*1000),
                    edgecolors='black', zorder=4, vmin=0, vmax=max(weights.max(), 0.1))

    # Afficher le score au dessus des défenseurs importants
    for i, w in enumerate(weights):
        if w > 0.10: # Seuil d'affichage
            ax.text(def_x[i], def_y[i]+1, f"{w:.2f}", fontsize=10, fontweight='bold', color='darkred')

    # Mise en page
    plt.colorbar(sc, label="Niveau de Pression (Attention Score)")
    ax.set_title(f"Analyse de la Pression Défensive (Action #{idx})", fontsize=14)
    ax.set_xlabel("Distance Relative X (Yards)")
    ax.set_ylabel("Distance Relative Y (Yards)")
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(0, color='black', alpha=0.2)
    ax.axvline(0, color='black', alpha=0.2)
    ax.legend(loc='lower right')

    plt.show()