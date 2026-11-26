# AttentionDTA: Viva & Presentation Defense Guide
**Group 17 | Statistical Learning (DSC4804) | BML Munjal University**

Use this guide to master your project presentation and answer any viva question with complete confidence.

---

### Core Concept 1: What is the problem we are solving?
* **Problem:** In drug discovery for diseases like cancer, screening millions of small-molecule chemical compounds against hundreds of human target proteins experimentally in a laboratory costs billions of dollars and takes decades.
* **Our Solution:** A statistical machine learning model called **AttentionDTA** that predicts **Drug–Target Binding Affinity (DTA)** directly from the drug's chemical sequence (**SMILES**) and the protein's amino acid sequence (**FASTA**).

---

### Core Concept 2: Why Cancer? What is the biological context?
* Our benchmarks focus on **protein kinases** (the **Davis dataset** has 442 human kinases and 68 kinase inhibitor drugs).
* **Why kinases?** Kinases control cell signaling and division. When kinases mutate or become hyperactive, they cause cancer cells to divide uncontrollably.
* **Examples of real cancer targets in our project:**
  1. **EGFR:** Overexpressed in **Non-Small Cell Lung Cancer (NSCLC)**. Target of **Gefitinib** and **Erlotinib**.
  2. **ABL1 / BCR-ABL:** The driver of **Chronic Myeloid Leukemia (CML)**. Target of **Imatinib (Gleevec)**.
  3. **BRAF:** Mutated (V600E) in >50% of **Melanomas**. Target of **Vemurafenib**.
  4. **ERBB2 / HER2:** Amplified in aggressive **Breast & Gastric Cancers**. Target of **Lapatinib**.

---

### Core Concept 3: Why DTA (Regression) instead of DTI (Classification)?
* **DTI (Drug-Target Interaction):** Binary classification (Does it bind? $1 = \text{Yes}, 0 = \text{No}$).
  * *Limitation:* It loses critical information on binding potency. A drug that binds weakly has the same "1" label as a life-saving potent drug. Also, negative pairs are hard to verify experimentally.
* **DTA (Drug-Target Affinity):** Continuous regression predicting physical binding metrics:
  * $pK_d = -\log_{10}(K_d)$ (Equilibrium Dissociation Constant in Molar units).
  * Higher $pK_d$ means tighter binding (e.g., $pK_d > 8.0 \implies K_d < 10\text{ nM}$, indicating a potent cancer inhibitor).

---

### Core Concept 4: What is the "Attention Mechanism" and why does it matter?
* Older models like **DeepDTA** used "late fusion":
  * Drug CNN and Protein CNN processed their inputs completely independently, and only concatenated their final vectors at the very end.
* **AttentionDTA introduces Two-Sided Multi-Head Attention:**
  * It computes an interaction attention matrix:
    $$A = \tanh\left(W_d D \cdot (W_p P)^T\right)$$
  * **Row-wise sum $\rightarrow$ Drug Attention Vector:** Highlights which chemical atoms/functional groups in the drug are actively locking into the target.
  * **Column-wise sum $\rightarrow$ Protein Attention Vector:** Highlights which amino acid residues in the kinase form the active binding pocket.
  * This answers the question: *"Which part of the cancer protein combines with which part of the drug?"*

---

### Core Concept 5: What was Group 17's experimental investigation? (The Cold-Start Finding)
* **Standard S1 Split Reproduction:**
  * Our reproduced model achieved an $R^2 \approx 0.6853$ (average of Fold 1 and Fold 2 out of 5 folds due to computational budget), directionally consistent with the published paper's 5-fold average of $R^2 = 0.7409$.
* **Drug-Disjoint Cold-Start Split:**
  * We split the dataset such that **14 drugs in the test set were never seen during training** (evaluated with a single random seed).
  * On the validation set (with known drugs), $R^2$ was **0.6953**.
  * But on the unseen test drugs, **$R^2$ collapsed to -0.0079** (essentially zero or predicting the dataset mean).
* **Viva Punchline:**
  * *"Our independent reproduction demonstrates that while AttentionDTA performs well on known chemical scaffolds, its attention mechanism struggles to generalize to genuinely novel drugs. In cold-start conditions, the apparent interaction learning is largely scaffold memorization, which is the key bottleneck in real-world computational oncology."*

---

### Top 10 Rapid-Fire Viva Questions & Answers

1. **Q: What is a SMILES string?**  
   *A:* Simplified Molecular Input Line Entry System—a standard ASCII text representation of chemical structures (e.g., `C` for carbon, `O` for oxygen, `=` for double bond).

2. **Q: What input lengths did you use?**  
   *A:* Padded or truncated to length **100** for drug SMILES strings and **1,200** for protein amino acid sequences, matching the AttentionDTA specification.

3. **Q: What are the convolutional layers doing?**  
   *A:* Three 1D-CNN layers with 32, 64, and 96 filters. They extract local n-gram patterns (e.g., amino acid motifs and chemical functional groups).

4. **Q: Why use multiple attention heads?**  
   *A:* Multi-head attention allows the model to jointly attend to information from different representation subspaces (e.g., hydrogen bonding vs. hydrophobic packing interactions).

5. **Q: What is $R^2$ and what does an $R^2 \le 0$ mean?**  
   *A:* Coefficient of Determination. $R^2 = 1$ is a perfect model; $R^2 = 0$ means the model performs no better than simply predicting the average affinity value of the training set. An $R^2 \le 0$ in cold-start means complete failure to generalize to unseen chemical structures.

6. **Q: What loss function was used during training?**  
   *A:* Mean Squared Error (MSE) loss, with Adam optimizer and early stopping (patience 50).

7. **Q: What is the Davis dataset?**  
   *A:* A benchmark bioactivity dataset containing 30,056 experimental binding interactions between 442 human protein kinases and 68 small-molecule inhibitor compounds.

8. **Q: What is the difference between S1 and S4 split in the paper?**  
   *A:* S1 is a random split (both drugs and proteins overlap between train and test). S4 is a full cold-start split where both drugs and proteins in the test set are completely unseen.

9. **Q: Why did our drug-disjoint experiment show $R^2 = -0.0079$ when the paper's S4 reported $0.0756$?**  
   *A:* Our test set had only 14 held-out drugs with single-seed evaluation, which increases variance. However, both results confirm the steep drop in generalization capability.

10. **Q: How does this help an oncologist or medicinal chemist?**  
    *A:* It acts as a rapid pre-screening filter to rank drug candidates before laboratory synthesis, while the attention heatmap pinpoints specific binding pocket mutations that might cause drug resistance.
