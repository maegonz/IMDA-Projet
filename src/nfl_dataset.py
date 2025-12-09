import numpy as np
import pandas as pd
import torch
import glob
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# 1. Classe Dataset Intelligente (Normalisation)
class NFLDataset(Dataset):
    def __init__(self, pt_file):
        print(f"Chargement du dataset : {pt_file} ...")
        data = torch.load(pt_file)
        
        self.queries = data['queries']
        self.keys = data['keys']
        self.labels = data['labels']
        
        # Vérification des dimensions pour être sûr
        print(f"    Chargé {len(self.labels)} exemples.")
        print(f"   - Query Shape : {self.queries.shape} (Doit être N, 90)")
        print(f"   - Keys Shape  : {self.keys.shape}   (Doit être N, 22, 9)")
        print(f"   - Labels Shape: {self.labels.shape} (Doit être N, 2)")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # On clone juste les tenseurs. 
        # AUCUN CALCUL ICI car tout est déjà fait dans la génération V4.
        q = self.queries[idx].clone()
        k = self.keys[idx].clone()
        y = self.labels[idx].clone()

        return q, k, y
    

# # 1. Classe Dataset Intelligente (Normalisation)
# class NFLDataset(Dataset):
#     def __init__(self, pt_file):
#         data = torch.load(pt_file)
#         self.queries = data['queries']
#         self.keys = data['keys']
#         self.labels = data['labels']

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         q = self.queries[idx].clone()
#         k = self.keys[idx].clone()
#         y = self.labels[idx].clone()

#         # --- NORMALISATION ---
#         # Query: [Speed, Dir, DistX, DistY]
#         q[0] /= 10.0   # Speed (max ~10)
#         q[1] /= 360.0  # Dir
#         q[2] /= 50.0   # DistX (max ~50)
#         q[3] /= 50.0   # DistY

#         # Keys: [RelX, RelY, Speed, Dir]
#         k[:, 0] /= 50.0
#         k[:, 1] /= 50.0
#         k[:, 2] /= 10.0
#         k[:, 3] /= 360.0

#         # Label: [dX, dY] (Déplacement sur 10 frames, env 10 yards max)
#         y /= 10.0

#         return q, k, y