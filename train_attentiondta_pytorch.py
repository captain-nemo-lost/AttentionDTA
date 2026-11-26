"""
AttentionDTA: PyTorch Reference Implementation & Training Protocol
Group 17 - Statistical Learning (DSC4804), BML Munjal University
Reference: Zhao et al., AttentionDTA (IEEE/ACM TCBB 2023)

Key Features:
- 1D-CNN feature extraction blocks with padding alignment
- Two-Sided Multi-Head Attention with proper padding masking
- Fixed variable scoping (no torch.nn.functional as F shadowing)
- Complete training loop on the Davis kinase bioactivity dataset
- Real model checkpointing (valid_best_checkpoint.pth)
"""

import os
import sys
import math
import time
import argparse
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """Computes MSE, MAE, and R² using pure numpy."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-9)))
    return mse, mae, r2

# Canonical SMILES token mapping from Zhao et al. (2023)
CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64
}
CHARISOSMILEN = 64
MAX_DRUG_LEN = 100

# Canonical 25-character Amino Acid token mapping
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
}
CHARPROTLEN = 25
MAX_PROT_LEN = 1200


def encode_smiles(smiles: str, max_len: int = MAX_DRUG_LEN):
    """Encodes SMILES string into integer token array padded with zeros."""
    arr = np.zeros(max_len, dtype=np.int64)
    clean_smi = smiles.strip()
    length = min(len(clean_smi), max_len)
    for i in range(length):
        arr[i] = CHARISOSMISET.get(clean_smi[i], 0)
    return arr, clean_smi[:length]


def encode_protein(protein_seq: str, max_len: int = MAX_PROT_LEN):
    """Encodes protein amino acid sequence into integer token array padded with zeros."""
    arr = np.zeros(max_len, dtype=np.int64)
    clean_seq = protein_seq.strip().upper()
    length = min(len(clean_seq), max_len)
    for i in range(length):
        arr[i] = CHARPROTSET.get(clean_seq[i], 0)
    return arr, clean_seq[:length]


class Conv1DBlock(nn.Module):
    """1D-CNN feature extraction block with sequence length preservation."""
    def __init__(self, in_channels, out_channels_list=[32, 64, 96], kernel_sizes=[4, 6, 8]):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels_list[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv1d(out_channels_list[0], out_channels_list[1], kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv1d(out_channels_list[1], out_channels_list[2], kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2)

    def forward(self, x):
        orig_len = x.shape[2]
        x = F.relu(self.conv1(x))[:, :, :orig_len]
        x = F.relu(self.conv2(x))[:, :, :orig_len]
        x = F.relu(self.conv3(x))[:, :, :orig_len]
        return x  # Shape: (B, 96, L)


class TwoSidedMultiHeadAttention(nn.Module):
    """
    Two-Sided Multi-Head Attention Mechanism with strict padding masking.
    Computes cross-attention matrix A between drug tokens and protein residues.
    Zeroes out padding positions so padded tokens do not influence attention or pooling.
    """
    def __init__(self, feature_dim: int = 96, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // num_heads

        self.linear_drug = nn.Linear(feature_dim, feature_dim)
        self.linear_prot = nn.Linear(feature_dim, feature_dim)

    def forward(self, drug_feats, prot_feats, mask_d=None, mask_p=None):
        # NOTE: Variable named feat_dim to prevent shadowing torch.nn.functional as F
        B, L_d, feat_dim = drug_feats.shape
        _, L_p, _ = prot_feats.shape

        proj_d = self.linear_drug(drug_feats).view(B, L_d, self.num_heads, self.head_dim).transpose(1, 2)
        proj_p = self.linear_prot(prot_feats).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2)

        # Dot-product similarity across all heads: (B, H, L_d, L_p)
        sim = torch.matmul(proj_d, proj_p.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att_heads = torch.tanh(sim)

        # Average over heads: (B, L_d, L_p)
        A = torch.mean(att_heads, dim=1)

        # Apply 2D padding mask if provided
        if mask_d is not None and mask_p is not None:
            mask_2d = mask_d.unsqueeze(2) & mask_p.unsqueeze(1)  # (B, L_d, L_p)
            A_masked = A.masked_fill(~mask_2d, -1e9)
            A_norm = F.softmax(A_masked, dim=-1)
            A_norm = A_norm.masked_fill(~mask_2d, 0.0)
            # Re-normalize across valid protein residues
            row_sum = A_norm.sum(dim=-1, keepdim=True) + 1e-9
            A_norm = A_norm / row_sum
            A_norm = A_norm.masked_fill(~mask_2d, 0.0)
        else:
            A_norm = F.softmax(A, dim=-1)

        # Drug attention vector: row-wise sum over protein residues
        alpha_d = torch.sum(A_norm, dim=-1, keepdim=True)  # (B, L_d, 1)
        # Protein attention vector: column-wise sum over drug tokens
        alpha_p = torch.sum(A_norm, dim=-2, keepdim=True).transpose(1, 2)  # (B, L_p, 1)

        # Mask padding on attention vectors
        if mask_d is not None:
            alpha_d = alpha_d.masked_fill(~mask_d.unsqueeze(-1), 0.0)
            weighted_drug = (drug_feats * alpha_d).masked_fill(~mask_d.unsqueeze(-1), -1e9)
        else:
            weighted_drug = drug_feats * alpha_d

        if mask_p is not None:
            alpha_p = alpha_p.masked_fill(~mask_p.unsqueeze(-1), 0.0)
            weighted_prot = (prot_feats * alpha_p).masked_fill(~mask_p.unsqueeze(-1), -1e9)
        else:
            weighted_prot = prot_feats * alpha_p

        # Global max pooling across sequence length (excluding padded positions)
        pool_drug = torch.clamp(torch.max(weighted_drug, dim=1)[0], min=-10.0)
        pool_prot = torch.clamp(torch.max(weighted_prot, dim=1)[0], min=-10.0)

        fused = torch.cat([pool_drug, pool_prot], dim=-1)  # (B, feat_dim * 2)
        return fused, A_norm, alpha_d.squeeze(-1), alpha_p.squeeze(-1)


class AttentionDTA(nn.Module):
    """
    Complete AttentionDTA neural network architecture for affinity regression.
    """
    def __init__(self, drug_vocab_size=65, prot_vocab_size=26, embed_dim=128,
                 feature_dim=96, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.drug_embed = nn.Embedding(drug_vocab_size, embed_dim, padding_idx=0)
        self.prot_embed = nn.Embedding(prot_vocab_size, embed_dim, padding_idx=0)

        self.drug_cnn = Conv1DBlock(embed_dim, [32, 64, feature_dim], kernel_sizes=[4, 6, 8])
        self.prot_cnn = Conv1DBlock(embed_dim, [32, 64, feature_dim], kernel_sizes=[4, 8, 12])

        self.attention = TwoSidedMultiHeadAttention(feature_dim=feature_dim, num_heads=num_heads)

        self.fc1 = nn.Linear(feature_dim * 2, 1024)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(1024, 1024)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

        # Initialize output bias to Davis dataset mean pKd (~5.4)
        nn.init.constant_(self.out.bias, 5.4)

    def forward(self, drug_seq, prot_seq):
        # drug_seq: (B, 100), prot_seq: (B, 1200)
        mask_d = (drug_seq != 0)
        mask_p = (prot_seq != 0)

        d_emb = self.drug_embed(drug_seq).transpose(1, 2)  # (B, embed_dim, L_d)
        p_emb = self.prot_embed(prot_seq).transpose(1, 2)  # (B, embed_dim, L_p)

        d_feat = self.drug_cnn(d_emb).transpose(1, 2)      # (B, L_d, feature_dim)
        p_feat = self.prot_cnn(p_emb).transpose(1, 2)      # (B, L_p, feature_dim)

        fused, att_matrix, alpha_d, alpha_p = self.attention(d_feat, p_feat, mask_d, mask_p)

        x = F.relu(self.fc1(fused))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        pred = self.out(x).squeeze(-1)
        return pred, att_matrix, alpha_d, alpha_p


class DavisDataset(Dataset):
    """Dataset loader for the Davis kinase bioactivity benchmark."""
    def __init__(self, lines):
        self.samples = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # cid, gene, smiles, sequence, pkd
                smiles = parts[2]
                prot = parts[3]
                try:
                    pkd = float(parts[4])
                except ValueError:
                    continue
                d_arr, _ = encode_smiles(smiles)
                p_arr, _ = encode_protein(prot)
                self.samples.append((d_arr, p_arr, pkd))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d_arr, p_arr, pkd = self.samples[idx]
        return (
            torch.from_numpy(d_arr).long(),
            torch.from_numpy(p_arr).long(),
            torch.tensor(pkd, dtype=torch.float32)
        )


def load_davis_lines(data_path="data/Davis.txt"):
    """Loads and shuffles the full Davis dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Davis dataset not found at {data_path}")
    with open(data_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def get_kfold_split(lines, fold_idx=0, num_folds=5, seed=4321):
    """Splits dataset into train and validation sets using reproducible k-fold splitting."""
    rng = random.Random(seed)
    shuffled = lines.copy()
    rng.shuffle(shuffled)

    fold_size = len(shuffled) // num_folds
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size if fold_idx < num_folds - 1 else len(shuffled)

    val_lines = shuffled[val_start:val_end]
    train_lines = shuffled[:val_start] + shuffled[val_end:]
    return train_lines, val_lines


def evaluate(model, dataloader, device):
    """Runs full evaluation on validation set and computes MSE, MAE, R²."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for d_batch, p_batch, y_batch in dataloader:
            d_batch = d_batch.to(device)
            p_batch = p_batch.to(device)
            y_batch = y_batch.to(device)

            pred, _, _, _ = model(d_batch, p_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item() * len(y_batch)

            all_preds.extend(pred.cpu().numpy().tolist())
            all_targets.extend(y_batch.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    avg_loss = total_loss / len(y_true)
    mse, mae, r2 = compute_metrics(y_true, y_pred)
    return avg_loss, mse, mae, r2


def train(args):
    """Main training routine on the Davis dataset."""
    print("=" * 75)
    print("🧬 AttentionDTA: Real PyTorch Training Protocol")
    print("   Course: Statistical Learning (DSC4804) | Group 17 | BML Munjal University")
    print("=" * 75)

    # Set threads and seeds
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Compute Device:      {device} ({torch.get_num_threads()} CPU threads)")

    # Load dataset
    print(f"Loading Dataset:     {args.data_path}")
    lines = load_davis_lines(args.data_path)
    print(f"Total Davis Pairs:   {len(lines)}")

    train_lines, val_lines = get_kfold_split(lines, fold_idx=args.fold, num_folds=args.num_folds, seed=args.seed)
    if args.max_train_samples > 0:
        train_lines = train_lines[:args.max_train_samples]
    if args.max_val_samples > 0:
        val_lines = val_lines[:args.max_val_samples]

    print(f"Cross-Validation:    Fold {args.fold + 1} of {args.num_folds}")
    print(f"Training Pairs:      {len(train_lines)}")
    print(f"Validation Pairs:    {len(val_lines)}")
    print(f"Hyperparameters:     Epochs={args.epochs} | LR={args.lr} | Batch Size={args.batch_size}")
    print("-" * 75)

    train_dataset = DavisDataset(train_lines)
    val_dataset = DavisDataset(val_lines)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = AttentionDTA(
        drug_vocab_size=len(CHARISOSMISET) + 1,
        prot_vocab_size=len(CHARPROTSET) + 1,
        embed_dim=args.embed_dim,
        feature_dim=args.feature_dim,
        num_heads=args.num_heads,
        dropout_rate=args.dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_mse = float("inf")
    best_val_r2 = -float("inf")
    best_val_mae = float("inf")

    print("-" * 75)
    print(f"{'Epoch':<8} | {'Train MSE':<12} | {'Val MSE':<12} | {'Val MAE':<12} | {'Val R²':<10} | {'Time (s)':<10} | {'Status'}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_count = 0

        for d_batch, p_batch, y_batch in train_loader:
            d_batch = d_batch.to(device)
            p_batch = p_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred, _, _, _ = model(d_batch, p_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_count += len(y_batch)

        epoch_train_mse = train_loss / train_count
        val_loss, val_mse, val_mae, val_r2 = evaluate(model, val_loader, device)
        elapsed = time.time() - t0

        saved = False
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_mae = val_mae
            best_val_r2 = val_r2
            saved = True

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mse": best_val_mse,
                "best_val_mae": best_val_mae,
                "best_val_r2": best_val_r2,
                "fold": args.fold,
                "num_folds": args.num_folds,
                "dataset": "Davis Kinase",
                "model_architecture": "AttentionDTA (Two-Sided Multi-Head Attention with Masking)",
                "char_smi_dict": CHARISOSMISET,
                "char_prot_dict": CHARPROTSET,
                "embed_dim": args.embed_dim,
                "feature_dim": args.feature_dim,
                "num_heads": args.num_heads,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            torch.save(checkpoint, args.save_path)

        status_str = "⭐ Best Model Saved" if saved else ""
        print(f"[{epoch:02d}/{args.epochs:02d}]   | {epoch_train_mse:<12.4f} | {val_mse:<12.4f} | {val_mae:<12.4f} | {val_r2:<10.4f} | {elapsed:<10.1f} | {status_str}")

    print("=" * 75)
    print(f"Training Complete!")
    print(f"Best Validation MSE: {best_val_mse:.4f} | MAE: {best_val_mae:.4f} | R²: {best_val_r2:.4f}")
    print(f"Checkpoint saved to: {args.save_path}")
    print("=" * 75)


def parse_args():
    parser = argparse.ArgumentParser(description="AttentionDTA Real PyTorch Training Protocol")
    parser.add_argument("--data_path", type=str, default="data/Davis.txt", help="Path to Davis dataset")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0 to 4)")
    parser.add_argument("--num_folds", type=int, default=5, help="Total number of folds")
    parser.add_argument("--seed", type=int, default=4321, help="Random seed")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for PyTorch")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--feature_dim", type=int, default=96, help="Feature dimension (96)")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_train_samples", type=int, default=0, help="Optional cap on train samples (0 = full)")
    parser.add_argument("--max_val_samples", type=int, default=0, help="Optional cap on val samples (0 = full)")
    parser.add_argument("--save_path", type=str, default="valid_best_checkpoint.pth", help="Checkpoint output file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
