"""
AttentionDTA: Fast Real PyTorch Training Script
Group 17 - Statistical Learning (DSC4804), BML Munjal University

This script executes real PyTorch training of AttentionDTA on the Davis kinase dataset
using PyTorch, calculating real MSE, MAE, and R² metrics at every epoch, and exporting
a real model checkpoint.

Usage:
  python quick_train.py --epochs 3 --batch_size 64
  python quick_train.py --max_train_samples 1000 --epochs 2
"""

import os
import sys
import argparse

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_attentiondta_pytorch import train


def main():
    parser = argparse.ArgumentParser(description="AttentionDTA Fast Real Training Execution")
    parser.add_argument("--data_path", type=str, default=os.path.join(PROJECT_ROOT, "data", "Davis.txt"),
                        help="Path to Davis dataset")
    parser.add_argument("--epochs", type=int, default=2, help="Number of real training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    parser.add_argument("--num_folds", type=int, default=5, help="Total folds")
    parser.add_argument("--seed", type=int, default=4321, help="Random seed")
    parser.add_argument("--threads", type=int, default=8, help="CPU threads")
    parser.add_argument("--embed_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--feature_dim", type=int, default=96, help="Feature dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_train_samples", type=int, default=1280,
                        help="Maximum training samples for fast execution (0 = full 19,965 pairs)")
    parser.add_argument("--max_val_samples", type=int, default=384,
                        help="Maximum validation samples for fast execution (0 = full 4,991 pairs)")
    parser.add_argument("--save_path", type=str, default=os.path.join(PROJECT_ROOT, "valid_best_checkpoint.pth"),
                        help="Checkpoint output path")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
