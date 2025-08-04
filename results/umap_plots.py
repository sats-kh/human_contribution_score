import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Define paths and constants
CSV_PATH = "./hc.csv"
OUT_DIR = "./umap"
RANDOM_SEED = 42

# UMAP parameter sets to iterate over
umap_param_sets = [
    {"n_neighbors": 4, "min_dist": 0.01},
]

os.makedirs(OUT_DIR, exist_ok=True)

# List of metrics to use for UMAP dimensionality reduction
metrics_for_umap = [
    "aesthetic_score", "metaclip_score", "perplexity_score", "semantic_divergence",
    "Q_k", "M_k", "perplexity_reduction", "semantic_similarity",
    "Mk_Qk", "Human_Contribution_Score", "Q0"
]

# Load and preprocess data
try:
    df_raw = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    raise SystemExit(f"[ERROR] '{CSV_PATH}' not found. Check file path.")

# Select relevant columns for UMAP
df_umap = df_raw[["prompt_sequence_number", "category_number"] + metrics_for_umap].copy()

# Handle missing values by filling with the mean of the column
for col in metrics_for_umap:
    if df_umap[col].isnull().sum() > 0:
        df_umap[col] = df_umap[col].fillna(df_umap[col].mean())

# Log-transform 'bert_ppl' if it exists to handle skewed data
if "bert_ppl" in metrics_for_umap:
    min_pos = df_umap["bert_ppl"][df_umap["bert_ppl"] > 0].min()
    min_pos = min_pos if pd.notna(min_pos) else 1e-9
    df_umap["bert_ppl"] = df_umap["bert_ppl"].apply(
        lambda x: np.log(x) if x > 0 else np.log(min_pos)
    )

print(f"[INFO] Data prepared. Shape: {df_umap.shape}")
print("[INFO] Scaling numerical features...")

# Set matplotlib style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.5,
})

# Define a set of custom colors for plotting
custom_colors = [
    "#4E79A7",  # Blue
    "#F28E2B",  # Orange
    "#E15759",  # Red
    "#76B7B2",  # Teal
    "#59A14F",  # Green
    "#EDC948",  # Yellow
    "#B07AA1",  # Purple
    "#FF9DA7",  # Pink
    "#9C755F",  # Brown
    "#BAB0AC"   # Gray
]

# Loop through each UMAP parameter set
for params in umap_param_sets:
    n_neighbors = params["n_neighbors"]
    min_dist = params["min_dist"]

    # Create step groups for color-coding the plot (3-step intervals)
    def create_step_group_by_3(step):
        group_start = ((step - 1) // 3) * 3 + 1
        group_end = group_start + 2
        return f"{group_start}-{group_end}"

    df_umap["step_group_3"] = df_umap["prompt_sequence_number"].apply(create_step_group_by_3)
    df_umap["step_group_3"] = pd.Categorical(
        df_umap["step_group_3"],
        categories=sorted(df_umap["step_group_3"].unique(), key=lambda x: int(x.split('-')[0])),
        ordered=True
    )

    # Balanced sampling (max 10000 samples per group) to prevent overplotting
    df_balanced = df_umap.groupby("step_group_3").apply(
        lambda x: x.sample(n=10000, random_state=RANDOM_SEED) if len(x) > 10000 else x
    ).reset_index(drop=True)

    # Scale features and apply UMAP
    X_balanced = df_balanced[metrics_for_umap]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)

    print(f"\n[PROCESS] Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")

    umap_model = UMAP(
        n_components=2,
        random_state=RANDOM_SEED,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        transform_seed=RANDOM_SEED,
    )
    umap_results = umap_model.fit_transform(X_scaled)

    df_balanced["umap-2d-x"] = umap_results[:, 0]
    df_balanced["umap-2d-y"] = umap_results[:, 1]

    # Create publication-quality plot
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(
        x="umap-2d-x",
        y="umap-2d-y",
        hue="step_group_3",
        palette=custom_colors[:df_balanced["step_group_3"].nunique()],
        data=df_balanced,
        legend="full",
        alpha=1,
        s=25,
        # edgecolor='none'
    )

    # Customize plot
    # plt.title(f"UMAP Projection (n_neighbors={n_neighbors}, min_dist={min_dist})", pad=20)
    plt.xlabel("")
    plt.ylabel("")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Move legend outside the plot
    legend = plt.legend(title="Step", loc='upper right', 
            borderaxespad=0.75, frameon=True, fontsize=12, title_fontsize=13)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    plt.tight_layout()

    # Save high-resolution figure
    fname = f"umap_plot_n{n_neighbors}_d{min_dist:.3f}.png"
    plt.savefig(os.path.join(OUT_DIR, fname), dpi=400, bbox_inches='tight')
    plt.close()
    print(f" ✔ Saved → {fname}")

print("\n[DONE] All UMAP plots with balanced groups generated.")