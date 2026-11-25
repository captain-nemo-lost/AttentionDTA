"""
AttentionDTA REST API Backend
Group 17 - Statistical Learning (DSC4804), BML Munjal University

Endpoints:
  GET  /api/health       - Health check
  GET  /api/targets      - Pre-loaded cancer kinase benchmarks (EGFR, ABL1, etc.)
  POST /api/predict      - Run AttentionDTA affinity regression & cross-attention extraction
  GET  /api/benchmarks   - S1 vs Cold-Start evaluation metrics table
"""

import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np

# Add project root to sys.path to import attention_dta_model
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from attention_dta_model import AttentionDTAInference

FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)  # Enable Cross-Origin Resource Sharing for standalone frontend clients

# Initialize inference engine
model_engine = AttentionDTAInference()

# Load cancer targets benchmark
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "davis_cancer_kinases.csv")
if os.path.exists(DATA_PATH):
    df_targets = pd.read_csv(DATA_PATH)
else:
    df_targets = pd.DataFrame()


@app.route("/")
def serve_index():
    """Serves the frontend dashboard index.html"""
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return jsonify({"message": "AttentionDTA API is running. Frontend folder not found."}), 200


@app.route("/<path:path>")
def serve_static(path):
    """Serves static assets (css, js, images)"""
    if os.path.exists(os.path.join(FRONTEND_DIR, path)):
        return send_from_directory(FRONTEND_DIR, path)
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "AttentionDTA Backend API",
        "group": "Group 17 (BML Munjal University)",
        "course": "Statistical Learning (DSC4804)"
    })


@app.route("/api/targets", methods=["GET"])
def get_targets():
    """Returns list of pre-configured cancer kinase target profiles."""
    if df_targets.empty:
        return jsonify([])
    records = df_targets.to_dict(orient="records")
    return jsonify(records)


@app.route("/api/benchmarks", methods=["GET"])
def get_benchmarks():
    """Returns the S1 vs Cold-Start experimental results comparison."""
    benchmarks = [
        {
            "condition": "Davis S1 (Random Split - Fold 1)",
            "source": "Our Reproduction",
            "mse": 0.2129,
            "mae": 0.2195,
            "r2": 0.6682,
            "notes": "Converged ~epoch 231"
        },
        {
            "condition": "Davis S1 (Random Split - Fold 2)",
            "source": "Our Reproduction",
            "mse": 0.2173,
            "mae": 0.2342,
            "r2": 0.7024,
            "notes": "Converged ~epoch 106"
        },
        {
            "condition": "Davis S1 (Average of Reproduced Folds)",
            "source": "Our Reproduction",
            "mse": 0.2151,
            "mae": 0.2268,
            "r2": 0.6853,
            "notes": "Directionally consistent with paper"
        },
        {
            "condition": "Davis S1 (Original Published Paper)",
            "source": "Zhao et al. (2023)",
            "mse": 0.1741,
            "mae": 0.2044,
            "r2": 0.7409,
            "notes": "5-fold cross-validation"
        },
        {
            "condition": "Davis Drug-Disjoint (Our Validation)",
            "source": "Our Experiment",
            "mse": 0.1879,
            "mae": 0.2013,
            "r2": 0.6953,
            "notes": "Training drugs still present"
        },
        {
            "condition": "Davis Drug-Disjoint (Our Held-Out Test)",
            "source": "Our Experiment",
            "mse": 0.8206,
            "mae": 0.5329,
            "r2": -0.0079,
            "notes": "14 novel unseen drugs (Generalization collapse)"
        },
        {
            "condition": "Davis S4 Full Cold-Start (Paper Reference)",
            "source": "Zhao et al. (2023)",
            "mse": 0.6058,
            "mae": 0.4731,
            "r2": 0.0756,
            "notes": "Both drugs and proteins held out"
        }
    ]
    return jsonify(benchmarks)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST payload:
    {
       "smiles": "COC1=...",
       "protein_seq": "MRPSG...",
       "mode": "standard" | "cold_start",
       "window_start": 0,
       "window_size": 60
    }
    """
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body provided"}), 400

    smiles = data.get("smiles", "").strip()
    protein_seq = data.get("protein_seq", "").strip()
    mode = data.get("mode", "standard")
    window_start = int(data.get("window_start", 0))
    window_size = int(data.get("window_size", 60))

    if not smiles or not protein_seq:
        return jsonify({"error": "Both 'smiles' and 'protein_seq' are required"}), 400

    try:
        raw_res = model_engine.predict(smiles, protein_seq, mode=mode)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    att_matrix = raw_res["attention_matrix"]
    alpha_drug = raw_res["alpha_drug"].tolist()
    alpha_prot = raw_res["alpha_prot"].tolist()
    valid_smiles = raw_res["valid_smiles"]
    valid_prot = raw_res["valid_prot"]

    # Slice window for crisp visual heatmap rendering
    prot_len = len(valid_prot)
    start = max(0, min(window_start, prot_len - 1))
    end = min(prot_len, start + window_size)

    display_att = att_matrix[:, start:end].tolist()
    display_prot_labels = [f"{valid_prot[i]}{i+1}" for i in range(start, end)]
    drug_labels = [f"{valid_smiles[i]}{i+1}" for i in range(len(valid_smiles))]

    response_payload = {
        "predicted_pKd": raw_res["predicted_pKd"],
        "Kd_nM": raw_res["Kd_nM"],
        "category": raw_res["category"],
        "color": raw_res["color"],
        "mode": raw_res["mode"],
        "valid_smiles": valid_smiles,
        "valid_prot": valid_prot,
        "drug_length": len(valid_smiles),
        "protein_length": prot_len,
        "top_residues": raw_res["top_residues"],
        "top_substructures": raw_res["top_substructures"],
        "heatmap": {
            "z": display_att,
            "x": display_prot_labels,
            "y": drug_labels,
            "window_start": start,
            "window_end": end,
            "total_protein_len": prot_len
        },
        "alpha_drug": alpha_drug,
        "alpha_prot": alpha_prot
    }

    return jsonify(response_payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting AttentionDTA REST API on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
