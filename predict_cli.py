"""
AttentionDTA: Command-Line Single-Pair Prediction & Interaction Analyzer
Group 17 - Statistical Learning (DSC4804), BML Munjal University

Usage:
  python predict_cli.py --condition standard
  python predict_cli.py --condition cold_start --target EGFR
  python predict_cli.py --smiles "COC1=..." --protein "MRPSG..."
"""

import argparse
import sys
import os
import pandas as pd
from attention_dta_model import AttentionDTAInference


def main():
    parser = argparse.ArgumentParser(description="AttentionDTA Cancer Kinase Affinity & Attention Predictor")
    parser.add_argument("--condition", choices=["standard", "cold_start"], default="standard",
                        help="Evaluation condition: 'standard' (S1 trained) or 'cold_start' (drug-disjoint)")
    parser.add_argument("--target", type=str, default="EGFR",
                        help="Pre-loaded target kinase name (e.g., EGFR, ABL1, BRAF, ERBB2, FLT3)")
    parser.add_argument("--smiles", type=str, default=None,
                        help="Custom drug SMILES string")
    parser.add_argument("--protein", type=str, default=None,
                        help="Custom target kinase amino acid sequence")

    args = parser.parse_args()

    # Load defaults from benchmark file if not provided
    csv_path = os.path.join(os.path.dirname(__file__), "data", "davis_cancer_kinases.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()

    smiles = args.smiles
    protein_seq = args.protein
    target_info = args.target

    if smiles is None or protein_seq is None:
        if not df.empty:
            matches = df[df["Target_Name"].str.upper() == args.target.upper()]
            if not matches.empty:
                row = matches.iloc[0]
                smiles = row["Drug_SMILES"]
                protein_seq = row["Protein_Sequence"]
                target_info = f"{row['Target_Name']} ({row['Disease_Context']}) with {row['Drug_Name']}"
            else:
                row = df.iloc[0]
                smiles = row["Drug_SMILES"]
                protein_seq = row["Protein_Sequence"]
                target_info = f"{row['Target_Name']} ({row['Disease_Context']}) with {row['Drug_Name']}"
        else:
            smiles = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"
            protein_seq = "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNY"
            target_info = "EGFR (Non-Small Cell Lung Cancer) with Gefitinib"

    print("=" * 70)
    print("🧬 AttentionDTA: Binding Affinity & Attention Interaction Engine")
    print("   Group 17 | Statistical Learning (DSC4804) | BML Munjal University")
    print("=" * 70)
    print(f"Target Profile:        {target_info}")
    print(f"Evaluation Condition:  {args.condition.upper()} ({'Known Chemical Space' if args.condition == 'standard' else 'Novel Unseen Drug (Cold-Start)'})")
    print(f"Drug SMILES Length:    {len(smiles)} characters")
    print(f"Protein Sequence Len:  {len(protein_seq)} amino acids")
    print("-" * 70)

    model = AttentionDTAInference()
    res = model.predict(smiles, protein_seq, mode=args.condition)

    if res.get("is_trained"):
        meta = res.get("checkpoint_meta", {})
        mse_val = meta.get("best_val_mse")
        mse_str = f"{mse_val:.4f}" if isinstance(mse_val, (int, float)) else str(mse_val)
        print(f"Model Checkpoint:      PyTorch AttentionDTA (Epoch {meta.get('epoch')}, Val MSE: {mse_str})")
    else:
        print("Model Checkpoint:      Initialized PyTorch AttentionDTA")

    print("\n[PREDICTION RESULTS]")
    print(f"  * Predicted Affinity (pK_d):  {res['predicted_pKd']:.2f}")
    print(f"  * Equilibrium Const (K_d):    {res['Kd_nM']:.2f} nM")
    print(f"  * Clinical Category:          {res['category']}")

    print("\n[TOP-3 PREDICTED CANCER KINASE BINDING RESIDUES]")
    for r in res["top_residues"][:3]:
        print(f"  - Position {r['position']}: Amino Acid '{r['residue']}' (Motif: ...{r['local_window']}...) | Attention: {r['attention_score']:.4f}")

    print("\n[TOP-3 PREDICTED ACTIVE DRUG SUBSTRUCTURES]")
    for d in res["top_substructures"][:3]:
        print(f"  - Index {d['index']}: Atom '{d['atom_char']}' (Fragment: {d['substructure']}) | Attention: {d['attention_score']:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
