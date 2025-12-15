import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from .nfl_seq2seq import NFLSeq2SeqModel
from .nfl_attention import NFLAttentionModel
from torch import nn, optim
from torch.utils.data import DataLoader


def train_model(model: NFLAttentionModel | NFLSeq2SeqModel,
                train_loader: DataLoader,
                criterion,
                optimizer: optim.Optimizer,
                device: torch.device,
                val_loader: DataLoader = None,
                num_epochs: int = 20):

    train_losses = []
    val_losses = []

    epoch_bar = tqdm(range(num_epochs), desc="Epochs")

    for epoch in epoch_bar:
        batch_bar = tqdm(train_loader, leave=False)
        model.train()
        total_loss = 0

        for q, k, y in batch_bar:
            optimizer.zero_grad()

            # Move to device
            q = q.to(device)
            k = k.to(device)
            y = y.to(device)

            if isinstance(model, NFLSeq2SeqModel):
                # for seq2seq models, prepare decoder input
                batch_size = y.size(0)
                start_token = torch.zeros(batch_size, 1, 2).to(device)
                decoder_input = torch.cat([start_token, y[:, :-1, :]], dim=1)
                preds = model(q, k, decoder_input)
            else:
                preds, _ = model(q, k)

            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_bar.set_postfix(batch_loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        epoch_bar.set_postfix(train_loss=avg_loss)

        if val_loader is not None:
            avg_val_loss = evaluate_model(model, val_loader, criterion, device)
            val_losses.append(avg_val_loss)
            epoch_bar.set_postfix(train_loss=avg_loss, val_loss=avg_val_loss)
        else:
            epoch_bar.set_postfix(train_loss=avg_loss)

    return train_losses, val_losses


def evaluate_model(model: NFLAttentionModel | NFLSeq2SeqModel,
                   test_loader: DataLoader,
                   criterion,
                   device: torch.device):

    model.eval()
    total_loss = 0

    with torch.no_grad():
        for q, k, y in test_loader:
            # Move to device
            q = q.to(device)
            k = k.to(device)
            y = y.to(device)

            if isinstance(model, NFLSeq2SeqModel):
                # for seq2seq models, prepare decoder input
                batch_size = y.size(0)
                start_token = torch.zeros(batch_size, 1, 2).to(device)
                decoder_input = torch.cat([start_token, y[:, :-1, :]], dim=1)
                preds = model(q, k, decoder_input)
            else:
                preds, _ = model(q, k)

            loss = criterion(preds, y)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)

    return avg_loss