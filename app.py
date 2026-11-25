"""
AttentionDTA: Cancer Kinase Binding Affinity & Attention Interaction Visualizer
Group 17 - Statistical Learning (DSC4804), BML Munjal University
Authors: Ayush Pratap Singh, Abhimanyu Sharma, Radhika Singh
Faculty: Dr. Arijit Maitra
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from attention_dta_model import AttentionDTAInference, MAX_DRUG_LEN, MAX_PROT_LEN

# Configure page
st.set_page_config(
    page_title="AttentionDTA | Cancer Kinase Visualizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished scientific styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .cancer-badge {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .highlight-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return AttentionDTAInference()


@st.cache_data
def load_cancer_benchmarks():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "davis_cancer_kinases.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


# Load model and benchmarks
model = load_model()
df_benchmarks = load_cancer_benchmarks()

# Header
st.markdown('<div class="main-title">🧬 AttentionDTA: Cancer Kinase Binding Affinity & Interaction Visualizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title"><b>Statistical Learning (DSC4804)</b> · Group 17 (Ayush Pratap Singh, Abhimanyu Sharma, Radhika Singh) · BML Munjal University</div>',
    unsafe_allow_html=True
)

# --- SIDEBAR: Configuration & Target Selection ---
st.sidebar.header("🎯 Target & Compound Selection")

cancer_options = [
    "EGFR — Non-Small Cell Lung Cancer (Gefitinib)",
    "EGFR — Non-Small Cell Lung Cancer (Erlotinib)",
    "ABL1 — Chronic Myeloid Leukemia (Imatinib / Gleevec)",
    "BRAF — Melanoma & Colorectal Cancer (Vemurafenib)",
    "ERBB2 / HER2 — Breast & Gastric Cancer (Lapatinib)",
    "FLT3 — Acute Myeloid Leukemia (Quizartinib)",
    "Custom Drug-Protein Input"
]

selected_case = st.sidebar.selectbox("Select Cancer Target Case Study", cancer_options)

# Load selected case data
if selected_case != "Custom Drug-Protein Input":
    case_idx = cancer_options.index(selected_case)
    row = df_benchmarks.iloc[case_idx]
    target_name = row["Target_Name"]
    disease_context = row["Disease_Context"]
    drug_name = row["Drug_Name"]
    default_smiles = row["Drug_SMILES"]
    default_seq = row["Protein_Sequence"]
    exp_pkd = float(row["Experimental_pKd"])
else:
    target_name = "Custom Kinase"
    disease_context = "Custom Cancer Profile"
    drug_name = "Custom Inhibitor"
    default_smiles = "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"
    default_seq = "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSHRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA"
    exp_pkd = 8.50

st.sidebar.markdown(f"**Target Protein:** `{target_name}`")
st.sidebar.markdown(f"**Disease Context:** *{disease_context}*")
st.sidebar.markdown(f"**Inhibitor Drug:** `{drug_name}`")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Evaluation Setting")
eval_mode = st.sidebar.radio(
    "Evaluation Protocol",
    ("Standard S1 Split (Known Chemical Space, R² ~ 0.69)",
     "Drug-Disjoint Cold-Start (Unseen Novel Drug, R² ~ -0.01)"),
    help="Toggle between standard random-split prediction and drug-disjoint cold-start evaluation to observe generalization behavior."
)
is_cold_start = "Cold-Start" in eval_mode
model_mode_code = "cold_start" if is_cold_start else "standard"

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Viva Tip:** In the S1 setting, the model has seen the drug family during training. "
    "In the Drug-Disjoint setting, the drug is completely unseen, exposing the generalization challenge."
)

# Input text boxes in an expander
with st.expander("📝 View / Edit Sequence & SMILES Inputs", expanded=(selected_case == "Custom Drug-Protein Input")):
    input_smiles = st.text_input("Drug SMILES (Ligand String):", value=default_smiles)
    input_protein = st.text_area("Target Kinase Sequence (Amino Acids):", value=default_seq, height=120)

# Run Inference
try:
    results = model.predict(input_smiles, input_protein, mode=model_mode_code)
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()

# --- MAIN DASHBOARD TABS ---
tab_overview, tab_attention, tab_substructures, tab_comparison = st.tabs([
    "📊 Binding Affinity & Results",
    "🔥 Two-Sided Attention Heatmap",
    "🎯 Binding Pockets & Pharmacophores",
    "⚖️ Cold-Start Robustness Comparison"
])

# ==================== TAB 1: OVERVIEW ====================
with tab_overview:
    st.subheader(f"Results for {target_name} ({disease_context}) with {drug_name}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Predicted Binding Affinity (pK_d)",
            value=f"{results['predicted_pKd']:.2f}",
            delta=f"{results['predicted_pKd'] - exp_pkd:+.2f} vs Exp ({exp_pkd:.2f})"
        )
    with col2:
        st.metric(
            label="Dissociation Constant (K_d)",
            value=f"{results['Kd_nM']:.2f} nM",
            help="Dissociation constant in nanomolar. Lower Kd indicates tighter binding."
        )
    with col3:
        st.metric(
            label="Binding Classification",
            value="Potent" if results['predicted_pKd'] >= 8.0 else ("Moderate" if results['predicted_pKd'] >= 7.0 else "Weak"),
            delta="Therapeutic Range" if results['predicted_pKd'] >= 7.5 else "Sub-therapeutic"
        )
    with col4:
        st.metric(
            label="Evaluation Condition",
            value="Cold-Start (Unseen)" if is_cold_start else "Standard S1",
            delta="Degraded Generalization" if is_cold_start else "High Confidence",
            delta_color="inverse" if is_cold_start else "normal"
        )

    st.markdown("---")
    
    col_desc, col_summary = st.columns([1.2, 0.8])
    with col_desc:
        st.markdown("### 🔬 Biological & Clinical Significance")
        st.markdown(f"""
        * **Cancer Target:** **{target_name}** is a critical receptor protein kinase driving tumor proliferation and survival in **{disease_context}**.
        * **Targeted Drug:** **{drug_name}** binds competitively into the ATP-binding pocket of {target_name}, inhibiting phosphorylation and blocking oncogenic downstream signaling cascades.
        * **Affinity Interpretation:** A predicted $pK_d$ of **{results['predicted_pKd']}** corresponds to an equilibrium dissociation constant $K_d$ of **{results['Kd_nM']} nM**. Values below 100 nM are typically considered active therapeutic candidates.
        """)
        
        st.markdown(f"""
        <div class="highlight-box">
            <b>Active Condition:</b> {eval_mode}<br>
            <i>{'⚠️ Warning: In cold-start mode, novel unseen drugs experience severe affinity regression collapse toward the dataset mean due to lack of shared chemical scaffolds during training.' if is_cold_start else '✅ In standard S1 mode, the attention mechanism effectively captures known kinase-ligand interaction patterns.'}</i>
        </div>
        """, unsafe_allow_html=True)

    with col_summary:
        st.markdown("### 📈 Affinity Score Distribution")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=results['predicted_pKd'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Binding Affinity (pK_d)", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [4, 11], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': results['color']},
                'steps': [
                    {'range': [4, 7.0], 'color': "#fee2e2"},
                    {'range': [7.0, 8.0], 'color': "#fef3c7"},
                    {'range': [8.0, 11], 'color': "#dcfce7"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.75,
                    'value': exp_pkd
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ==================== TAB 2: ATTENTION HEATMAP ====================
with tab_attention:
    st.subheader("🔥 Two-Sided Cross-Attention Interaction Matrix")
    st.markdown(
        "This interactive heatmap visualizes the **pairwise attention scores** computed by AttentionDTA's multi-head attention module. "
        "Each cell indicates how strongly a specific chemical character of the drug attends to an amino acid residue of the target cancer protein."
    )

    att_matrix = results["attention_matrix"]
    drug_chars = list(results["valid_smiles"])
    prot_chars = list(results["valid_prot"])

    # Provide a window slider for large protein sequences
    prot_len = len(prot_chars)
    if prot_len > 80:
        st.markdown("##### 🔍 Protein Residue Focus Window")
        win_col1, win_col2 = st.columns([1, 3])
        with win_col1:
            win_size = st.selectbox("Window Size (Residues)", [40, 60, 80, 100], index=1)
        with win_col2:
            max_start = max(0, prot_len - win_size)
            # Default to active kinase region (around position 600-800 if available, else 0)
            default_start = min(680, max_start) if prot_len > 700 else 0
            start_pos = st.slider("Select Protein Region (Residue Start Position)", 0, max_start, default_start, step=10)
        end_pos = min(prot_len, start_pos + win_size)
        display_prot = prot_chars[start_pos:end_pos]
        display_prot_labels = [f"{ch}{start_pos + i + 1}" for i, ch in enumerate(display_prot)]
        display_att = att_matrix[:, start_pos:end_pos]
    else:
        display_prot = prot_chars
        display_prot_labels = [f"{ch}{i+1}" for i, ch in enumerate(prot_chars)]
        display_att = att_matrix

    drug_labels = [f"{ch}{i+1}" for i, ch in enumerate(drug_chars)]

    # Heatmap with Plotly
    fig_heatmap = px.imshow(
        display_att,
        x=display_prot_labels,
        y=drug_labels,
        labels=dict(x="Cancer Protein Residue Position", y="Drug Chemical Token (SMILES)", color="Attention Weight"),
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    fig_heatmap.update_layout(
        height=550,
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis_title="Target Cancer Kinase Residues",
        yaxis_title="Drug SMILES Substructures"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.caption("Hover over any cell to see the exact residue, drug character, and cross-attention weight.")

# ==================== TAB 3: BINDING POCKETS & PHARMACOPHORES ====================
with tab_substructures:
    st.subheader("🎯 Critical Binding Sites & Active Drug Substructures")
    st.markdown(
        "By applying **row-wise and column-wise summation** on the attention matrix, AttentionDTA decomposes the interaction into "
        "independent 1D importance profiles for both the cancer protein and the drug molecule."
    )

    col_prot, col_drug = st.columns(2)

    with col_prot:
        st.markdown("#### 1. Cancer Protein: Predicted Binding Pocket Residues")
        alpha_p = results["alpha_prot"]
        
        # Line chart of protein attention
        fig_prot_line = go.Figure()
        fig_prot_line.add_trace(go.Scatter(
            x=list(range(1, len(alpha_p) + 1)),
            y=alpha_p,
            mode='lines',
            line=dict(color='#2563EB', width=1.8),
            name='Residue Importance'
        ))
        fig_prot_line.update_layout(
            title="Protein Sequence Importance Profile",
            xaxis_title="Amino Acid Position (1 to L)",
            yaxis_title="Normalized Attention Weight",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_prot_line, use_container_width=True)

        st.markdown("**Top-5 Predicted Key Binding Residues:**")
        df_top_residues = pd.DataFrame(results["top_residues"])
        df_top_residues.columns = ["Position", "Residue", "Local Window Motif", "Attention Score"]
        st.dataframe(df_top_residues, use_container_width=True, hide_index=True)
        st.caption("Residues with high attention align with the catalytic kinase cleft, hinge region, and ATP-binding pocket.")

    with col_drug:
        st.markdown("#### 2. Drug Molecule: Pharmacophore Substructures")
        alpha_d = results["alpha_drug"]
        
        fig_drug_bar = go.Figure()
        fig_drug_bar.add_trace(go.Bar(
            x=list(range(1, len(alpha_d) + 1)),
            y=alpha_d,
            marker_color='#10B981',
            name='Substructure Importance'
        ))
        fig_drug_bar.update_layout(
            title="Drug Chemical Token Importance Profile",
            xaxis_title="SMILES Character Index (1 to 100)",
            yaxis_title="Normalized Attention Weight",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_drug_bar, use_container_width=True)

        st.markdown("**Top-5 Predicted Active Chemical Fragments:**")
        df_top_drug = pd.DataFrame(results["top_substructures"])
        df_top_drug.columns = ["Index", "Atom / Symbol", "Chemical Fragment", "Attention Score"]
        st.dataframe(df_top_drug, use_container_width=True, hide_index=True)
        st.caption("Active fragments correspond to hydrogen-bonding heterocycles (quinazolines, pyrimidines) and lipophilic anchors.")

# ==================== TAB 4: COLD-START COMPARISON ====================
with tab_comparison:
    st.subheader("⚖️ Generalization & Cold-Start Robustness Analysis")
    st.markdown(
        "A central objective of Group 17's research paper was investigating the **reproducibility and cold-start robustness** of AttentionDTA. "
        "Here we compare the reproduced experimental results against the published literature on the Davis benchmark dataset."
    )

    # Experimental numbers from attentiondta_results_summary.md.pdf
    comparison_data = {
        "Evaluation Condition / Split": [
            "Davis S1 (Random Split - Fold 1)",
            "Davis S1 (Random Split - Fold 2)",
            "Davis S1 (Average of Reproduced Folds)",
            "Davis S1 (Original Paper - Zhao et al. 2023)",
            "Davis Drug-Disjoint (Our Validation Set)",
            "Davis Drug-Disjoint (Our Held-Out Test Set)",
            "Davis S4 Full Cold-Start (Paper Reference)"
        ],
        "Source": [
            "Our Reproduction", "Our Reproduction", "Our Reproduction",
            "Zhao et al. (Published)", "Our Experiment", "Our Experiment", "Zhao et al. (Published)"
        ],
        "MSE (↓)": [0.2129, 0.2173, 0.2151, 0.1741, 0.1879, 0.8206, 0.6058],
        "MAE (↓)": [0.2195, 0.2342, 0.2268, 0.2044, 0.2013, 0.5329, 0.4731],
        "R² Score (↑)": [0.6682, 0.7024, 0.6853, 0.7409, 0.6953, -0.0079, 0.0756],
        "Status / Notes": [
            "Converged ~epoch 231", "Converged ~epoch 106", "Consistent with paper (gap ~0.055)",
            "5-fold cross-validation benchmark", "Train drugs in held-out pairs",
            "14 completely unseen drugs (Collapse)", "Both drugs & proteins held out"
        ]
    }
    df_comp = pd.DataFrame(comparison_data)

    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    col_chart, col_takeaway = st.columns([1.2, 0.8])
    with col_chart:
        # Comparison bar chart of R2
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            x=["S1 Repro (Avg)", "S1 Paper", "Drug-Disjoint Val", "Drug-Disjoint Test", "S4 Full Cold-Start"],
            y=[0.6853, 0.7409, 0.6953, -0.0079, 0.0756],
            marker_color=['#3B82F6', '#1E40AF', '#10B981', '#EF4444', '#F59E0B'],
            text=["0.685", "0.741", "0.695", "-0.008", "0.076"],
            textposition='auto'
        ))
        fig_r2.update_layout(
            title="R² Performance Across Evaluation Settings on Davis Dataset",
            yaxis_title="R² Score (Higher is better, ≤0 means no predictive skill)",
            height=340,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with col_takeaway:
        st.markdown("#### 🔑 Key Scientific Insights")
        st.markdown("""
        1. **Reproducibility Verified:** Our S1 reproduction achieves $R^2 = 0.6853$, showing directional consistency with the published $0.7409$.
        2. **The Cold-Start Collapse:** When tested on 14 completely unseen drugs, $R^2$ plunges from **0.695 to -0.0079**.
        3. **Why this matters for Cancer Research:** Models with high standard accuracy often fail in real-world virtual screening because novel chemical leads do not resemble training molecules.
        """)


st.markdown("---")
st.caption("Developed for Statistical Learning (DSC4804) Evaluation · Group 17 · BML Munjal University")
