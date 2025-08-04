import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# 1) Path and Constant Definitions
# ──────────────────────────────────────────────────────────────────────────────
CSV_PATH = "./hc.csv"
OUT_DIR = "./contribution_metrics"
X_AXIS_MAX_STEP = 25  # Max step for the X-axis
SMOOTH_WINDOW = 6     # Rolling average window size (in steps)

os.makedirs(OUT_DIR, exist_ok=True)

# Metrics configuration (labels are adjusted for LaTeX font rendering)
metrics_to_plot = {
    "semantic_divergence":  {"label": r"$\text{Semantic Divergence } (W_k)$", "apply_log": False, "division_factor": 1},
    "Q_k": {"label": r"$\text{Image Quality } (Q_k)$", "apply_log": False, "division_factor": 1},
    "perplexity_reduction": {"label": r"$\text{Perplexity Reduction } (\Delta \text{PPL}_k$)", "apply_log": False, "division_factor": 1},
    "M_k": {"label": r"$\text{Modification Strength } (M_k)$", "apply_log": False, "division_factor": 1},
    "Human_Contribution_Score": {"label": r"$\text{Human Contribution Score } (HC)$", "apply_log": False, "division_factor": 1},
}

# ──────────────────────────────────────────────────────────────────────────────
# 2) Visualization Style
# ──────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "text.usetex": True,                       # Enable LaTeX rendering
    "font.family": "serif",                    # Set font family to serif
    "font.serif": ["Times New Roman"],         # Specify Times New Roman font
    'font.size': 10,                           # Overall base font size
    'axes.labelsize': 10,                      # X and Y axis label size
    'axes.titlesize': 10,                      # Title size (no explicit title in this plot)
    'legend.fontsize': 9,                      # Legend font size
    'xtick.labelsize': 9,                      # X-axis tick label size
    'ytick.labelsize': 9,                      # Y-axis tick label size
    'figure.dpi': 400,                         # Screen DPI
    'savefig.dpi': 400,                        # DPI for saved figures
    'axes.linewidth': 0.8,                     # Axis border line width
    'grid.linewidth': 0.4,                     # Grid line width
    'grid.alpha': 0.5,                         # Grid transparency
    "figure.constrained_layout.use": True,     # Use constrained_layout for auto-adjusting margins
    "text.latex.preamble": r"\usepackage{amsmath}" # Load amsmath for \text command
})
print(f"[INFO] Loading input CSV: {CSV_PATH}")

# ──────────────────────────────────────────────────────────────────────────────
# 3) Data Loading and Basic Preprocessing
# ──────────────────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    raise SystemExit(f"[ERROR] '{CSV_PATH}' not found. Check file path.")

required_base_columns = ["category_number", "prompt_sequence_number", "Q0"]
required_metric_columns = list(metrics_to_plot.keys())
missing = [c for c in (required_base_columns + required_metric_columns) if c not in df.columns]
if missing:
    raise ValueError(f"[ERROR] Missing required columns: {missing}")

# Type conversion and filtering
df["category_number"] = pd.to_numeric(df["category_number"], errors="coerce")
df = df.dropna(subset=["category_number"])
df["category_number"] = df["category_number"].astype(int)
df = df[df["category_number"] >= 1]

# Filter data up to the maximum X-axis step
df = df[df["prompt_sequence_number"] <= X_AXIS_MAX_STEP].copy()
if df.empty:
    raise SystemExit(f"[ERROR] No data with prompt_sequence_number ≤ {X_AXIS_MAX_STEP}")

print(f"[INFO] Data loaded. Shape after basic filtering: {df.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 4) Plotting Loop
# ──────────────────────────────────────────────────────────────────────────────
# Metrics that should be plotted starting from Step 2 (as Step 1 values are 0/NaN)
columns_starting_from_step2 = [
    "new M_k", "delta_bert_ppl", "delta_bert_ppl_square", "delta_bert_ppl_abs",
    "new_M_k_abs_ppl",
    "delta_log_bert_ppl", "delta_log_bert_ppl_square", "delta_log_bert_ppl_abs",
    "new_M_k_log_ppl_square", "new_M_k_log_ppl_abs",
    "delta Q_k",
]

for metric_col, cfg in metrics_to_plot.items():
    metric_label = cfg["label"]
    apply_log = cfg["apply_log"]
    division_factor = cfg["division_factor"]

    print(f"\n[PROCESS] {metric_label} ({metric_col})")

    # ── Prepare Values ────────────────────────────────────────────────────────────
    df_plot = df.copy()
    df_plot["Metric_Value_For_Plotting"] = df_plot[metric_col] / division_factor

    if metric_col in ["metaclip_score", "aesthetic_score", "bert_ppl"]:
        df_plot.loc[df_plot["Metric_Value_For_Plotting"] < 0, "Metric_Value_For_Plotting"] = np.nan

    if apply_log:
        min_pos = df_plot["Metric_Value_For_Plotting"][df_plot["Metric_Value_For_Plotting"] > 0].min()
        min_pos = min_pos if pd.notna(min_pos) else 1e-9
        df_plot["Metric_Value_For_Plotting"] = df_plot["Metric_Value_For_Plotting"].apply(
            lambda x: np.log(x) if x > 0 else np.nan if pd.isna(x) else np.log(min_pos)
        )
        metric_label += r" (\text{log-scale})"

    df_plot = df_plot.dropna(subset=["Metric_Value_For_Plotting"])
    if df_plot.empty:
        print("  ↳ No valid data. Skipping.")
        continue

    # ── Calculate Step-wise Mean & Standard Error ──────────────────────────────────
    grouped_stats = (
        df_plot.groupby("prompt_sequence_number")["Metric_Value_For_Plotting"]
            .agg(
                mean_value='mean',
                sem='sem' # Calculate Standard Error of the Mean
            )
            .reset_index()
    )

    if 1 in grouped_stats["prompt_sequence_number"].values:
        step1_val = grouped_stats.loc[grouped_stats["prompt_sequence_number"] == 1, "mean_value"].iloc[0]
        if (metric_col in columns_starting_from_step2) and (pd.isna(step1_val) or np.isclose(step1_val, 0.0)):
            grouped_stats = grouped_stats[grouped_stats["prompt_sequence_number"] > 1]
    
    if grouped_stats.empty:
        print("  ↳ Insufficient post-filter data. Skipping.")
        continue

    # ── Apply Moving Average (Smoothing) ──────────────────────────────────────────
    grouped_stats["smoothed_value"] = (
        grouped_stats["mean_value"]
        .rolling(window=SMOOTH_WINDOW, center=True, min_periods=1)
        .mean()
    )
    
    # ── Visualization ────────────────────────────────────────────────────────────
    # Adjust figure size (e.g., for a paper's double-column width)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Define font properties
    xlabel_font = FontProperties(family="Times New Roman", size=10)
    ylabel_font = FontProperties(family="Times New Roman", size=10)
    xtick_font = FontProperties(family="Times New Roman", size=9)
    ytick_font = FontProperties(family="Times New Roman", size=9)

    # (1) 2nd order polynomial regression line
    sns.regplot(
        x="prompt_sequence_number", 
        y="smoothed_value",
        data=grouped_stats, 
        scatter=False, 
        ci=None,
        order=2,
        color="red",
        line_kws={"lw": 2, "alpha": 0.9}, 
        ax=ax
    )
    ax.errorbar(
        x=grouped_stats["prompt_sequence_number"],
        y=grouped_stats["mean_value"],
        fmt='o',
        color="darkblue",
        ecolor="darkblue",
        markersize=6,
        zorder=8
    )

    # Add plot spines/borders
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['top'].set_edgecolor('black')
    ax.spines['right'].set_edgecolor('black')
    ax.spines['left'].set_edgecolor('black')
    ax.spines['bottom'].set_edgecolor('black')

    # (3) Axes, Labels, Grid
    ax.set_xticks(range(0, X_AXIS_MAX_STEP + 1, 5))
    
    # Apply FontProperties to labels and ticks
    ax.set_xlabel(r"Step", fontproperties=xlabel_font)
    ax.set_ylabel(r"{}".format(metric_label), fontproperties=ylabel_font)
    
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(xtick_font)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(ytick_font)

    ax.set_xlim(0, X_AXIS_MAX_STEP + 1)
    
    ax.grid(axis="y", linestyle="-", alpha=0.3, color="gray")
    ax.grid(axis="x", linestyle="-", alpha=0.3, color="gray")

    fig.tight_layout() # Call tight_layout to remove extra whitespace

    # ── Save ─────────────────────────────────────────────────────────────
    fname = (
        f"{metric_col}_sequence_mean_curved_regression_plot.pdf"
        .replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    )
    # Save as a PDF with tight bounding box to remove whitespace
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=400, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ Saved → {fname}")

print("\n[DONE] All plots generated.")