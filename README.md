# AttentionDTA: Drug–Target Binding Affinity Prediction & Attention Visualizer

**Course:** Statistical Learning (DSC4804), 7th Semester B.Tech CSE  
**Institution:** BML Munjal University, Gurugram  
**Faculty In-Charge:** Dr. Arijit Maitra  
**Group:** Group 17  
**Team Members:** Ayush Pratap Singh, Abhimanyu Sharma, Radhika Singh  

---

## 1. Project Overview

Targeted cancer therapies rely on small-molecule inhibitor drugs binding tightly to specific disease-causing proteins called **kinases** (such as EGFR in Lung Cancer, ABL1 in Leukemia, or BRAF in Melanoma).

This project implements and evaluates **AttentionDTA**, a deep learning model that predicts continuous binding affinity ($pK_d$) directly from raw chemical strings (**SMILES**) and protein sequences (**amino acids**), using a **two-sided multi-head attention mechanism**.

Crucially, this project addresses the core question:
> **"Which part of the cancer protein interacts with which part of the drug molecule?"**

The attention mechanism produces an interpretable **Drug × Protein Cross-Attention Matrix**, highlighting:
1. **Predicted Binding Pockets:** Specific amino acid residues of the cancer kinase that form contact interfaces.
2. **Active Pharmacophore Fragments:** Specific functional groups of the drug that contribute most to affinity.

All attention maps and predictions in the interactive dashboard are computed via an authentic PyTorch neural network loading trained model weights from `valid_best_checkpoint.pth` with sequence padding strictly masked.

---

## 2. Key Findings: Reproduction & Cold-Start Robustness

Our study reproduced AttentionDTA on the standard **Davis kinase dataset** (30,056 experimental binding interactions across 442 kinases and 68 inhibitor drugs) and investigated the **generalization challenge**:

| Evaluation Setting | Source | MSE (↓) | MAE (↓) | $R^2$ Score (↑) | Biological / ML Significance |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Davis S1 (Random Split, Avg)** | **Our Reproduction** | **0.2151** | **0.2268** | **0.6853** | Directionally consistent with published paper ($0.7409$). |
| **Davis S1 (Full 5-fold)** | Zhao et al. (2023) | 0.1741 | 0.2044 | 0.7409 | Published benchmark baseline. |
| **Davis Drug-Disjoint (Val)** | **Our Experiment** | **0.1879** | **0.2013** | **0.6953** | High accuracy when drugs are in training set. |
| **Davis Drug-Disjoint (Test)** | **Our Experiment** | **0.8206** | **0.5329** | **-0.0079** | **Generalization Collapse:** Fails on 14 novel unseen drugs. |
| **Davis S4 (Full Cold-Start)** | Zhao et al. (2023) | 0.6058 | 0.4731 | 0.0756 | Both drugs & proteins held out (Paper reference). |

### Important Methodological Caveats
* **2-of-5-Folds Caveat:** Due to local hardware compute constraints, our reproduced Davis S1 random-split metrics reflect the average of 2 cross-validation folds (Fold 1: $R^2 = 0.6682$, Fold 2: $R^2 = 0.7024$) out of the standard 5 folds, whereas the published literature baseline in Zhao et al. (2023) evaluated across all 5 complete folds.
* **Single-Seed Caveat:** The drug-disjoint cold-start evaluation ($R^2 = -0.0079$) was conducted using a single random seed with 14 held-out novel drugs in the test set. Single-seed splits with smaller hold-out sample sizes have higher variance than multi-seed cross-validation. However, the acute performance drop confirms the core finding that sequence attention models risk memorizing chemical scaffolds rather than learning universal physical binding dynamics.
* **Sequence Verification:** Protein kinase sequences in the benchmark dataset (including BRAF and FLT3) are verified against canonical UniProt entries (**BRAF: P15056**, **FLT3: P36888**).

---

## 3. How to Run the Project

### A. Launching the Interactive Web Dashboard
Open your terminal inside this folder and run:
```bash
./.venv/bin/streamlit run app.py
```
This opens the browser dashboard (`http://localhost:8501`) featuring:
* Real-time PyTorch inference with live cross-attention heatmap extraction.
* Padding-masked attention visualization (unpadded drug SMILES tokens × cancer protein residues).
* Pre-loaded cancer targets (EGFR, ABL1, BRAF, HER2, FLT3) and custom input support.
* 1D active site residue importance and pharmacophore token profiles.
* Cold-start generalization analysis and model architecture diagrams.

### B. Launching the Flask REST API & Web Client
```bash
PORT=5050 ./.venv/bin/python backend/server.py
```
Serves the web client and endpoints at `http://localhost:5050`:
* `GET /api/health` — Health check
* `GET /api/targets` — Pre-loaded cancer targets
* `GET /api/benchmarks` — Published and experimental benchmark table
* `POST /api/predict` — PyTorch model inference & masked attention extraction

### C. Running Predictions via Command Line
```bash
# Standard prediction on EGFR with Gefitinib
./.venv/bin/python predict_cli.py --condition standard --target EGFR

# Prediction on BRAF with Vemurafenib (canonical UniProt P15056 sequence)
./.venv/bin/python predict_cli.py --condition standard --target BRAF

# Custom drug SMILES and protein sequence
./.venv/bin/python predict_cli.py --smiles "COC1=CC=CC=C1" --protein "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNY"
```

### D. Training the PyTorch Model on the Davis Dataset
```bash
# Full training on Davis dataset (5 folds available, reproducible with seed 4321)
./.venv/bin/python train_attentiondta_pytorch.py --epochs 5 --batch_size 128 --threads 8

# Fast training execution on a sample
./.venv/bin/python quick_train.py --epochs 3 --batch_size 64
```

---

## 4. Repository Structure

```
AttentionDTA/
├── app.py                         # Interactive Streamlit Web Application (Real PyTorch model)
├── attention_dta_model.py         # PyTorch inference engine with masked cross-attention
├── train_attentiondta_pytorch.py  # PyTorch model definition & complete training pipeline
├── predict_cli.py                 # CLI prediction tool for single pairs
├── quick_train.py                 # Fast real PyTorch training execution script
├── README.md                      # Project documentation with methodological caveats
├── VIVA_AND_PROJECT_GUIDE.md      # Detailed viva questions, answers & statistical learning theory
├── valid_best_checkpoint.pth      # Real PyTorch trained model checkpoint (state_dict)
├── backend/
│   └── server.py                  # Flask REST API backend
├── frontend/
│   ├── index.html                 # Modern web dashboard HTML5 interface
│   ├── app.js                     # Interactive dashboard controller
│   └── style.css                  # Responsive stylesheet
├── data/
│   ├── Davis.txt                  # Full Davis kinase dataset (24,956 drug-protein pairs)
│   ├── davis_cancer_kinases.csv   # Pre-loaded cancer benchmarks (UniProt verified)
│   └── davis_dataset_sample.csv   # Sample Davis dataset pairs
└── Docs/
    ├── SLGroup_17.pdf             # Group 17 research & reproduction paper
    ├── AttentionDTA_Presentation (1).pptx # Group 17 presentation slides
    └── Statistical_Learning_DSC4804_2026_Course_Handout.pdf
```
