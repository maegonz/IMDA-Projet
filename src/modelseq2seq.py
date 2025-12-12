import torch
import torch.nn as nn
import math

# Aide pour dire au modèle l'ordre des frames (1, 2, 3...)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        return x + self.pe[:, :x.size(1)]

class NFLSeq2SeqModel(nn.Module):
    def __init__(self, receiver_dim=90, defender_dim=9, embed_dim=64, num_heads=4, num_decoder_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # --- PARTIE ENCODEUR (Similaire à avant) ---
        self.receiver_embedding = nn.Linear(receiver_dim, embed_dim)
        self.defender_embedding = nn.Linear(defender_dim, embed_dim)
        # Attention "Interne" (Le receveur regarde les défenseurs)
        self.encoder_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # --- PARTIE DÉCODEUR ---
        # On projette le point (dx, dy) en vecteur 64
        self.decoder_input_embedding = nn.Linear(2, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Le Décodeur Transformer standard
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Tête de prédiction finale
        self.output_head = nn.Linear(embed_dim, 2) # Sort (dx, dy)

    def generate_square_subsequent_mask(self, sz):
        # Masque pour empêcher de regarder le futur pendant l'entrainement
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src_rec, src_def, tgt_seq):
        """
        src_rec: Historique Receveur (Batch, 90)
        src_def: Contexte Défenseurs (Batch, 22, 9)
        tgt_seq: Séquence cible future (Batch, T_future, 2) -> Ce qu'on doit apprendre à prédire
        """
        # 1. ENCODAGE
        # Query = Le receveur (Batch, 1, 64)
        query = self.receiver_embedding(src_rec).unsqueeze(1)
        # Keys = Les défenseurs (Batch, 22, 64)
        keys = self.defender_embedding(src_def)
        
        # Le receveur 'calcule' son contexte tactique
        memory, _ = self.encoder_attention(query, keys, keys)
        # memory est maintenant le "résumé tactique" (Batch, 1, 64)
        
        # 2. DÉCODAGE (Training avec "Teacher Forcing")
        # On prépare l'entrée du décodeur (Embeddings + Position)
        tgt_emb = self.decoder_input_embedding(tgt_seq)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Création du masque causal (pour ne pas tricher en regardant la frame T+1 quand on est à T)
        seq_len = tgt_seq.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt_seq.device)
        
        # Le Décodeur regarde : 
        # A) La séquence cible passée (tgt_emb)
        # B) La mémoire de l'encodeur (memory)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        # 3. PRÉDICTION
        prediction = self.output_head(output) # (Batch, Seq_Len, 2)
        return prediction

    def predict_future(self, src_rec, src_def, num_steps=10):
        """
        Fonction spéciale pour l'INFÉRENCE (Visualisation)
        Génère frame par frame sans connaitre la réponse.
        """
        self.eval()
        with torch.no_grad():
            # 1. Encodage (identique)
            query = self.receiver_embedding(src_rec).unsqueeze(1)
            keys = self.defender_embedding(src_def)
            memory, _ = self.encoder_attention(query, keys, keys)
            
            # 2. Boucle de Génération
            # On commence avec un déplacement nul (0, 0)
            current_input = torch.zeros(src_rec.size(0), 1, 2).to(src_rec.device)
            predictions = []
            
            for _ in range(num_steps):
                # Préparer l'entrée
                tgt_emb = self.decoder_input_embedding(current_input)
                tgt_emb = self.pos_encoder(tgt_emb)
                
                # Le décodeur prédit la suite
                # Pas besoin de masque ici car on avance pas à pas
                out = self.decoder(tgt_emb, memory)
                
                # On prend juste le dernier point prédit
                next_point = self.output_head(out[:, -1, :]).unsqueeze(1) # (Batch, 1, 2)
                
                predictions.append(next_point)
                
                # RÉCURSIVITÉ : Le point prédit devient l'entrée de la prochaine étape
                current_input = torch.cat([current_input, next_point], dim=1)
            
            # On colle tout (Batch, num_steps, 2)
            return torch.cat(predictions, dim=1)