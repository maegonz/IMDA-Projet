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