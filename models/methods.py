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