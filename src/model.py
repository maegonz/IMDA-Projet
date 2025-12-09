import torch
import torch.nn as nn
import torch.nn.functional as F

class NFLAttentionModel(nn.Module):
    def __init__(self, receiver_dim=90, defender_dim=9, embed_dim=128, num_heads=4):
        super().__init__()
        
        self.receiver_embedding = nn.Linear(receiver_dim, embed_dim)
        self.defender_embedding = nn.Linear(defender_dim, embed_dim)
        self.dropout = nn.Dropout(0.2) # 20% de chances d'oublier (régularisation)
        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, receiver_data, defenders_data):
        # receiver_data: (Batch, 90)
        # defenders_data: (Batch, 22, 9)
        
        query = self.receiver_embedding(receiver_data).unsqueeze(1)
        keys = self.defender_embedding(defenders_data)
        values = keys 
        
        attn_output, attn_weights = self.attention(query, keys, values)
        x = self.dropout(attn_output.squeeze(1)) # Appliquer le dropout après l'attention
        prediction = self.regressor(x)
        return prediction, attn_weights
    

# class NFLAttentionModel(nn.Module):
#     def __init__(self, receiver_dim=4, defender_dim=4, embed_dim=64, num_heads=4):
#         super().__init__()

#         # 1. Embeddings Spécifiques (C'est la nouveauté)
#         # On projette le Receveur dans l'espace latent
#         self.receiver_embedding = nn.Linear(receiver_dim, embed_dim)

#         # On projette les Défenseurs dans le MÊME espace latent
#         self.defender_embedding = nn.Linear(defender_dim, embed_dim)

#         # 2. Multi-Head Attention (Le Cœur)
#         # batch_first=True pour avoir (Batch, Seq, Features)
#         self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

#         # 3. Tête de prédiction (Régression)
#         # On prédit la "Distance au ballon" (puisque c'est ce qu'on a mis dans le dataset)
#         # Si la distance prédite est petite, c'est que l'action est un succès.
#         self.regressor = nn.Sequential(
#             nn.Linear(embed_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 2)
#             # Pas de Sigmoid car on prédit une distance (en yards), pas une proba 0-1
#         )

#     def forward(self, receiver_data, defenders_data):
#         # receiver_data: (Batch, 4) -> (Vitesse, Dir, DistBallX, DistBallY)
#         # defenders_data: (Batch, 11, 4) -> (RelX, RelY, Vitesse, Dir)

#         # A. Projection (Embeddings)
#         # On ajoute une dimension "Séquence=1" au receveur pour qu'il devienne (Batch, 1, Embed)
#         query = self.receiver_embedding(receiver_data).unsqueeze(1)

#         # Les défenseurs sont déjà une séquence de 11, donc (Batch, 11, Embed)
#         keys = self.defender_embedding(defenders_data)
#         values = keys # Les valeurs sont les mêmes que les clés (Standard)

#         # B. Calcul de l'Attention
#         # attn_weights sera de forme (Batch, 1, 11)
#         # C'est TA MÉTRIQUE : L'importance de chaque défenseur
#         attn_output, attn_weights = self.attention(query, keys, values)

#         # C. Prédiction
#         # On retire la dimension séquence pour passer dans le régresseur
#         prediction = self.regressor(attn_output.squeeze(1))

#         return prediction, attn_weights