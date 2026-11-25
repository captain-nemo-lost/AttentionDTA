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

---

## 2. Key Findings: Reproduction & Cold-Start Robustness

Our study reproduced AttentionDTA on the standard **Davis kinase dataset** and investigated the **generalization challenge**:

| Evaluation Setting | Source | MSE (↓) | MAE (↓) | $R^2$ Score (↑) | Biological / ML Significance |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Davis S1 (Random Split, Avg)** | **Our Reproduction** | **0.2151** | **0.2268** | **0.6853** | Directionally consistent with published paper ($0.7409$). |
| **Davis S1 (Full 5-fold)** | Zhao et al. (2023) | 0.1741 | 0.2044 | 0.7409 | Published benchmark baseline. |
| **Davis Drug-Disjoint (Val)** | **Our Experiment** | **0.1879** | **0.2013** | **0.6953** | High accuracy when drugs are in training set. |
| **Davis Drug-Disjoint (Test)** | **Our Experiment** | **0.8206** | **0.5329** | **-0.0079** | **Generalization Collapse:** Fails on 14 novel unseen drugs. |
| **Davis S4 (Full Cold-Start)** | Zhao et al. (2023) | 0.6058 | 0.4731 | 0.0756 | Both drugs & proteins held out (Paper reference). |

**Main Takeaway:** When tested on known drug families, the attention mechanism performs well ($R^2 \approx 0.69$). But under a **drug-disjoint cold-start split**, performance drops to near-zero ($R^2 \approx -0.01$). This shows that sequence attention models risk memorizing chemical scaffolds rather than learning universal physical binding dynamics.

---

## 3. How to Run the Project

### A. Launching the Interactive Web Dashboard (Recommended for Demo)
Open your terminal inside this folder and run:
```bash
streamlit run app.py
```
This opens the browser dashboard (`http://localhost:8501`) featuring:
* Pre-loaded cancer targets (EGFR, ABL1, BRAF, HER2, FLT3) and custom inputs.
* Predicted Binding Affinity ($pK_d$) & Dissociation Constant ($K_d$ in nM).
* **Interactive 2D Cross-Attention Heatmap** (with hover-over atomic & residue detail).
* **1D Active Site / Pharmacophore Profiles**.
* **Cold-Start Generalization Comparison**.
* Complete Model Flowchart & Viva Notes.

### B. Running Single-Pair Predictions via Command Line
```bash
# Standard prediction on EGFR with Gefitinib
python predict_cli.py --condition standard --target EGFR

# Cold-start prediction to demonstrate generalization degradation
python predict_cli.py --condition cold_start --target EGFR

# Custom drug SMILES and protein sequence
python predict_cli.py --smiles "COC1=CC=CC=C1" --protein "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNY"
```

---

## 4. Repository Structure

```
Stat/
├── app.py                         # Interactive Streamlit Web Application
├── attention_dta_model.py         # AttentionDTA model architecture & inference engine
├── predict_cli.py                 # CLI prediction script for single pairs
├── train_attentiondta_pytorch.py  # PyTorch model definition & training protocol
├── README.md                      # This documentation file
├── VIVA_AND_PROJECT_GUIDE.md      # Detailed viva questions, answers & concepts
├── data/
│   ├── davis_cancer_kinases.csv   # Pre-loaded cancer benchmarks (EGFR, ABL1, etc.)
│   └── davis_dataset_sample.csv   # Sample Davis dataset pairs with split tags
├── SLGroup_17.pdf                 # Group 17 research & review paper
├── AttentionDTA_Presentation (1).pptx # Group 17 presentation slides
└── attentiondta_results_summary.md.pdf # Results summary & demo plan
```
