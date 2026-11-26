"""
AttentionDTA: Real Neural Network Inference Engine
Group 17 - Statistical Learning (DSC4804), BML Munjal University

This module implements the production inference pipeline for AttentionDTA:
1. Canonical SMILES & Protein sequence tokenization with padding (100 and 1200 tokens)
2. Loads trained PyTorch model state_dict from valid_best_checkpoint.pth
3. Executes real forward pass through 1D-CNNs, Two-Sided Multi-Head Attention, and MLP
4. Extracts genuine cross-attention interaction matrix with padding strictly masked
5. Computes continuous binding affinity pKd and dissociation constant Kd (nM)
"""

import os
import sys
import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train_attentiondta_pytorch import (
    AttentionDTA,
    CHARISOSMISET,
    CHARPROTSET,
    MAX_DRUG_LEN,
    MAX_PROT_LEN,
    encode_smiles,
    encode_protein
)


class AttentionDTAInference:
    """
    Inference engine that loads the trained PyTorch AttentionDTA weights
    and executes the authentic forward pass with padding masked.
    """

    def __init__(self, checkpoint_path=None, device=None):
        if checkpoint_path is None:
            checkpoint_path = os.path.join(PROJECT_ROOT, "valid_best_checkpoint.pth")
        self.checkpoint_path = checkpoint_path

        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        else:
            self.device = torch.device(device)

        self.model = AttentionDTA(
            drug_vocab_size=len(CHARISOSMISET) + 1,
            prot_vocab_size=len(CHARPROTSET) + 1,
            embed_dim=128,
            feature_dim=96,
            num_heads=4
        ).to(self.device)

        self.is_trained = False
        self.checkpoint_meta = {}

        if os.path.exists(self.checkpoint_path):
            try:
                ckpt = torch.load(self.checkpoint_path, map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    self.checkpoint_meta = {
                        "epoch": ckpt.get("epoch", "N/A"),
                        "best_val_mse": ckpt.get("best_val_mse", "N/A"),
                        "best_val_mae": ckpt.get("best_val_mae", "N/A"),
                        "best_val_r2": ckpt.get("best_val_r2", "N/A"),
                        "dataset": ckpt.get("dataset", "Davis Kinase"),
                        "saved_at": ckpt.get("saved_at", "N/A")
                    }
                    self.is_trained = True
                elif isinstance(ckpt, dict):
                    self.model.load_state_dict(ckpt)
                    self.is_trained = True
            except Exception as e:
                print(f"[Warning] Could not load checkpoint from {self.checkpoint_path}: {e}")

        self.model.eval()

    def predict(self, smiles: str, protein_seq: str, mode: str = "standard"):
        """
        Runs real PyTorch inference on drug SMILES and target protein sequence.
        Returns predicted pKd, Kd (nM), and unpadded attention matrix & vectors.
        """
        enc_drug, valid_smiles = encode_smiles(smiles, max_len=MAX_DRUG_LEN)
        enc_prot, valid_prot = encode_protein(protein_seq, max_len=MAX_PROT_LEN)

        L_d = len(valid_smiles)
        L_p = len(valid_prot)

        if L_d == 0 or L_p == 0:
            raise ValueError("Input SMILES or Protein Sequence cannot be empty.")

        d_tensor = torch.from_numpy(enc_drug).long().unsqueeze(0).to(self.device)
        p_tensor = torch.from_numpy(enc_prot).long().unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_pred, att_2d, alpha_d_full, alpha_p_full = self.model(d_tensor, p_tensor)

        pred_val = float(raw_pred.squeeze().item())
        predicted_pKd = round(float(np.clip(pred_val, 4.0, 11.5)), 2)
        kd_nm = round(float(10 ** (9 - predicted_pKd)), 2)

        # Slice attention outputs strictly to valid (unpadded) sequence lengths
        # Padding tokens have been masked inside the attention layer
        att_matrix = att_2d[0, :L_d, :L_p].cpu().numpy().astype(np.float32)
        alpha_drug = alpha_d_full[0, :L_d].cpu().numpy().astype(np.float32)
        alpha_prot = alpha_p_full[0, :L_p].cpu().numpy().astype(np.float32)

        # Normalize valid slice so sum equals 1 for interpretability
        if alpha_drug.sum() > 0:
            alpha_drug = alpha_drug / alpha_drug.sum()
        if alpha_prot.sum() > 0:
            alpha_prot = alpha_prot / alpha_prot.sum()

        # Binding classification category based on continuous pKd
        if predicted_pKd >= 8.0:
            category = "High Affinity / Potent Inhibitor (Kd < 10 nM)"
            color = "#00cc66"
        elif predicted_pKd >= 7.0:
            category = "Moderate Affinity (10 nM ≤ Kd < 100 nM)"
            color = "#ffaa00"
        else:
            category = "Weak or Non-Specific Binding (Kd ≥ 100 nM)"
            color = "#ff4444"

        # Identify top-5 most attended protein residue positions
        top_prot_indices = np.argsort(alpha_prot)[::-1][:min(5, L_p)]
        top_residues = []
        for idx in top_prot_indices:
            start = max(0, idx - 2)
            end = min(L_p, idx + 3)
            segment = valid_prot[start:end]
            top_residues.append({
                "position": int(idx + 1),
                "residue": valid_prot[idx],
                "local_window": segment,
                "attention_score": round(float(alpha_prot[idx]), 4)
            })

        # Identify top-5 most attended drug chemical tokens
        top_drug_indices = np.argsort(alpha_drug)[::-1][:min(5, L_d)]
        top_substructures = []
        for idx in top_drug_indices:
            start = max(0, idx - 2)
            end = min(L_d, idx + 3)
            token = valid_smiles[start:end]
            top_substructures.append({
                "index": int(idx + 1),
                "atom_char": valid_smiles[idx],
                "substructure": token,
                "attention_score": round(float(alpha_drug[idx]), 4)
            })

        # Identify top-8 pairwise lock-and-key contact interactions
        flat_att = att_matrix.flatten()
        top_flat_indices = np.argsort(flat_att)[::-1][:min(8, flat_att.size)]
        top_pairs = []
        for rank, f_idx in enumerate(top_flat_indices):
            r = int(f_idx // L_p)
            c = int(f_idx % L_p)
            d_atom = valid_smiles[r]
            p_res = valid_prot[c]
            d_frag = valid_smiles[max(0, r - 2):min(L_d, r + 3)]
            p_motif = valid_prot[max(0, c - 2):min(L_p, c + 3)]
            top_pairs.append({
                "rank": rank + 1,
                "drug_atom": f"{d_atom}{r + 1}",
                "drug_idx": r + 1,
                "drug_char": d_atom,
                "drug_fragment": d_frag,
                "prot_residue": f"{p_res}{c + 1}",
                "prot_pos": c + 1,
                "prot_char": p_res,
                "prot_motif": p_motif,
                "score": round(float(att_matrix[r, c]), 5)
            })

        return {
            "predicted_pKd": predicted_pKd,
            "Kd_nM": kd_nm,
            "category": category,
            "color": color,
            "attention_matrix": att_matrix,
            "alpha_drug": alpha_drug,
            "alpha_prot": alpha_prot,
            "valid_smiles": valid_smiles,
            "valid_prot": valid_prot,
            "top_residues": top_residues,
            "top_substructures": top_substructures,
            "top_pairs": top_pairs,
            "mode": mode,
            "checkpoint_meta": self.checkpoint_meta,
            "is_trained": self.is_trained
        }
