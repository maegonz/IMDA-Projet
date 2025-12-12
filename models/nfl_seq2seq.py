import torch
import torch.nn as nn
import math

"""Transformer-based Seq2Seq model for NFL trajectory prediction.

This module defines:
- PositionalEncoding: sinusoidal positional signals added to decoder inputs.
- NFLSeq2SeqModel: encoder-decoder architecture where the receiver attends to
    defender context, and a transformer decoder predicts a sequence of (dx, dy).
"""

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    max_len : int, default 50
        Maximum sequence length supported for positional cache.
    """
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Add positional encoding to a batch of sequences.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, seq_len, dim).

        Returns
        -------
        torch.Tensor
            Tensor of the same shape with positional signal added.
        """
        # Expect x shape: (Batch, Seq_Len, Dim)
        return x + self.pe[:, :x.size(1)]

class NFLSeq2SeqModel(nn.Module):
    """
    Encoder-decoder model using attention over defender context.

    The encoder embeds the receiver's historical sequence as a single query
    and all defenders in the current frame as keys/values. The decoder then
    predicts a sequence of displacements given previous targets and the
    encoded context.

    Parameters
    ----------
    receiver_dim : int, default 90
        Flattened feature size for the receiver history (HISTORY_SIZE * 9).
    defender_dim : int, default 9
        Feature size per defender in the context frame.
    embed_dim : int, default 64
        Model embedding dimension for all projections.
    num_heads : int, default 4
        Number of attention heads in encoder/decoder.
    num_decoder_layers : int, default 2
        Number of transformer decoder layers.
    """
    def __init__(self, receiver_dim=90, defender_dim=9, embed_dim=64, num_heads=4, num_decoder_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # --- ENCODER --- maps receiver history and defender context into embeddings
        self.receiver_embedding = nn.Linear(receiver_dim, embed_dim)  # (B, 90) -> (B, 64)
        self.defender_embedding = nn.Linear(defender_dim, embed_dim)  # (B, 22, 9) -> (B, 22, 64)
        # Receiver attends over defender context to build a tactical summary
        self.encoder_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # --- DECODER --- consumes past target displacements
        # Project each (dx, dy) target step into the model dimension
        self.decoder_input_embedding = nn.Linear(2, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        
        # Standard Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Final linear head to output displacement vectors
        self.output_head = nn.Linear(embed_dim, 2)  # (dx, dy)

    def generate_square_subsequent_mask(self, sz):
        """Create a causal mask preventing attention to future positions.

        Parameters
        ----------
        sz : int
            Sequence length for which to build the mask.

        Returns
        -------
        torch.Tensor
            Square mask of shape (sz, sz) for causal decoding.
        """
        # Stop the model from looking ahead in the target sequence
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src_rec, src_def, tgt_seq):
        """Run encoder-decoder to predict displacement sequence.

        Parameters
        ----------
        src_rec : torch.Tensor
            Receiver history features of shape (batch, 90).
        src_def : torch.Tensor
            Defender context features of shape (batch, 22, 9).
        tgt_seq : torch.Tensor
            Target displacement sequence of shape (batch, T_future, 2).

        Returns
        -------
        torch.Tensor
            Predicted displacement sequence of shape (batch, T_future, 2).
        """
        # 1) ENCODER
        # Query = receiver embedding (Batch, 1, 64)
        query = self.receiver_embedding(src_rec).unsqueeze(1)
        # Keys/Values = defender embeddings (Batch, 22, 64)
        keys = self.defender_embedding(src_def)
        
        # MultiheadAttention expects (B, Seq, Dim); here query has Seq=1, keys Seq=22
        memory, _ = self.encoder_attention(query, keys, keys)
        # memory is the tactical summary (Batch, 1, 64)
        
        # 2) DECODER (teacher forcing during training)
        # Prepare decoder inputs: embed target steps and add positional encoding
        tgt_emb = self.decoder_input_embedding(tgt_seq)  # (B, T, 2) -> (B, T, 64)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Build causal mask so step t cannot see future steps
        seq_len = tgt_seq.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt_seq.device)
        
        # Decoder attends over past target embeddings and encoder memory
        # TransformerDecoder expects tgt as (B, T, D) and memory as (B, S, D)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
        # 3) PREDICTION
        prediction = self.output_head(output)  # (Batch, Seq_Len, 2)
        return prediction