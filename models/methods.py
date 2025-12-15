import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

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


# def predict_future(self, src_rec, src_def, num_steps=10):
#     """
#     Fonction spéciale pour l'INFÉRENCE (Visualisation)
#     Génère frame par frame sans connaitre la réponse.
#     """
#     self.eval()
#     with torch.no_grad():
#         # 1. Encodage (identique)
#         query = self.receiver_embedding(src_rec).unsqueeze(1)
#         keys = self.defender_embedding(src_def)
#         memory, _ = self.encoder_attention(query, keys, keys)
        
#         # 2. Boucle de Génération
#         # On commence avec un déplacement nul (0, 0)
#         current_input = torch.zeros(src_rec.size(0), 1, 2).to(src_rec.device)
#         predictions = []
        
#         for _ in range(num_steps):
#             # Préparer l'entrée
#             tgt_emb = self.decoder_input_embedding(current_input)
#             tgt_emb = self.pos_encoder(tgt_emb)
            
#             # Le décodeur prédit la suite
#             # Pas besoin de masque ici car on avance pas à pas
#             out = self.decoder(tgt_emb, memory)
            
#             # On prend juste le dernier point prédit
#             next_point = self.output_head(out[:, -1, :]).unsqueeze(1) # (Batch, 1, 2)
            
#             predictions.append(next_point)
            
#             # RÉCURSIVITÉ : Le point prédit devient l'entrée de la prochaine étape
#             current_input = torch.cat([current_input, next_point], dim=1)
        
#         # On colle tout (Batch, num_steps, 2)
#         return torch.cat(predictions, dim=1)