"""
AttentionDTA: PyTorch Reference Implementation & Training Protocol
Group 17 - Statistical Learning (DSC4804)

This script provides the complete PyTorch definition of AttentionDTA for reference,
training replication, and model checkpointing as detailed in:
- Zhao et al., AttentionDTA (IEEE/ACM TCBB 2023)
- Group 17 Report: SLGroup_17.pdf
"""

import math
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class Conv1DBlock(nn.Module):
        def __init__(self, in_channels, out_channels_list=[32, 64, 96], kernel_sizes=[4, 6, 8]):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels_list[0], kernel_size=kernel_sizes[0], padding="same")
            self.conv2 = nn.Conv1d(out_channels_list[0], out_channels_list[1], kernel_size=kernel_sizes[1], padding="same")
            self.conv3 = nn.Conv1d(out_channels_list[1], out_channels_list[2], kernel_size=kernel_sizes[2], padding="same")

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            return x  # Shape: (B, 96, L)

    class TwoSidedMultiHeadAttention(nn.Module):
        def __init__(self, feature_dim=96, num_heads=4):
            super().__init__()
            self.num_heads = num_heads
            self.feature_dim = feature_dim
            self.head_dim = feature_dim // num_heads

            self.linear_drug = nn.Linear(feature_dim, feature_dim)
            self.linear_prot = nn.Linear(feature_dim, feature_dim)

        def forward(self, drug_feats, prot_feats):
            # drug_feats: (B, L_d, F), prot_feats: (B, L_p, F)
            B, L_d, F = drug_feats.shape
            _, L_p, _ = prot_feats.shape

            proj_d = self.linear_drug(drug_feats).view(B, L_d, self.num_heads, self.head_dim).transpose(1, 2)
            proj_p = self.linear_prot(prot_feats).view(B, L_p, self.num_heads, self.head_dim).transpose(1, 2)

            # Dot-product similarity across all heads
            sim = torch.matmul(proj_d, proj_p.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att_heads = torch.tanh(sim)  # (B, H, L_d, L_p)

            # Average over heads
            A = torch.mean(att_heads, dim=1)  # (B, L_d, L_p)

            # Softmax row-wise and column-wise
            A_norm = F.softmax(A, dim=-1)
            alpha_d = torch.sum(A_norm, dim=-1, keepdim=True)  # (B, L_d, 1)
            alpha_p = torch.sum(A_norm, dim=-2, keepdim=True).transpose(1, 2)  # (B, L_p, 1)

            # Reweight features
            weighted_drug = drug_feats * alpha_d
            weighted_prot = prot_feats * alpha_p

            # Global max pooling
            pool_drug = torch.max(weighted_drug, dim=1)[0]
            pool_prot = torch.max(weighted_prot, dim=1)[0]

            return torch.cat([pool_drug, pool_prot], dim=-1), A_norm

    class AttentionDTA(nn.Module):
        def __init__(self, drug_vocab_size=65, prot_vocab_size=26, embed_dim=128, feature_dim=96, num_heads=4):
            super().__init__()
            self.drug_embed = nn.Embedding(drug_vocab_size, embed_dim, padding_idx=0)
            self.prot_embed = nn.Embedding(prot_vocab_size, embed_dim, padding_idx=0)

            self.drug_cnn = Conv1DBlock(embed_dim, [32, 64, feature_dim], kernel_sizes=[4, 6, 8])
            self.prot_cnn = Conv1DBlock(embed_dim, [32, 64, feature_dim], kernel_sizes=[4, 8, 12])

            self.attention = TwoSidedMultiHeadAttention(feature_dim=feature_dim, num_heads=num_heads)

            self.fc1 = nn.Linear(feature_dim * 2, 1024)
            self.dropout1 = nn.Dropout(0.1)
            self.fc2 = nn.Linear(1024, 1024)
            self.dropout2 = nn.Dropout(0.1)
            self.fc3 = nn.Linear(1024, 512)
            self.out = nn.Linear(512, 1)

        def forward(self, drug_seq, prot_seq):
            # drug_seq: (B, 100), prot_seq: (B, 1200)
            d_emb = self.drug_embed(drug_seq).transpose(1, 2)  # (B, E, L_d)
            p_emb = self.prot_embed(prot_seq).transpose(1, 2)  # (B, E, L_p)

            d_feat = self.drug_cnn(d_emb).transpose(1, 2)      # (B, L_d, 96)
            p_feat = self.prot_cnn(p_emb).transpose(1, 2)      # (B, L_p, 96)

            fused, att_matrix = self.attention(d_feat, p_feat)

            x = F.relu(self.fc1(fused))
            x = self.dropout1(x)
            x = F.relu(self.fc2(x))
            x = self.dropout2(x)
            x = F.relu(self.fc3(x))
            pred = self.out(x)
            return pred.squeeze(-1), att_matrix
