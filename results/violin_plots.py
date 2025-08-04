import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter

# Ignore specific warnings to keep the output clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# 1) Path and Constant Definitions
# ──────────────────────────────────────────────────────────────────────────────
# Path to the input CSV file
CSV_PATH      = "./hc.csv"
# Directory to save the output plots
OUT_DIR       = "./violin_plots"

# Create the output directory if it doesn't exist
os.makedirs(OUT_DIR, exist_ok=True)

# List of metrics to visualize, with their labels
metrics_to_plot = {
    "metaclip_score": {"label": "CLIP Score"},
    "aesthetic_score": {"label": "Aesthetic Score"},
    "perplexity_score": {"label": "Perplexity"}
}

# Define bins and labels for prompt length categorization
bins = [0, 5, 15, 25, np.inf]
labels = ['Subject', 'Title', 'Summary', 'Detailed']
bucket_order = labels # Specify the order of buckets for the X-axis
palette = dict(zip(bucket_order, sns.color_palette("Set2", len(bucket_order))))

# ──────────────────────────────────────────────────────────────────────────────
# 2) Visualization Style (Optimized for publication)
# ──────────────────────────────────────────────────────────────────────────────
# Set a clean seaborn theme
sns.set_theme(style="white")
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "font.family": "Times New Roman",
    "figure.constrained_layout.use": True
})

print(f"[INFO] Loading input CSV: {CSV_PATH}")

# ──────────────────────────────────────────────────────────────────────────────
# 3) Data Loading, Grouping, and Filtering
# ──────────────────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    raise SystemExit(f"[ERROR] '{CSV_PATH}' not found. Check file path.")

# Define the required columns for the analysis
required_columns = ["prompt_length", "metaclip_score", "aesthetic_score", "perplexity_score"]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"[ERROR] Missing required columns: {missing}. Please check the CSV file.")

# Create prompt length groups based on the defined bins
df['prompt_length_group'] = pd.cut(df['prompt_length'], bins=bins, labels=labels, right=True)
# Drop rows where the prompt length group could not be determined (NaN)
df = df.dropna(subset=['prompt_length_group'])

# Make a copy of the dataframe for filtering outliers
df_filtered = df.copy()
print(f"\n[INFO] Data shape before outlier removal: {df_filtered.shape}")

# Loop through each metric to remove outliers
# This code removes the bottom 0.1% of outliers for each metric.
for metric in metrics_to_plot.keys():
    # Drop any NaN values for the current metric to get valid data
    valid_values = df_filtered[metric].dropna()
    if not valid_values.empty:
        # Calculate the 0.1th percentile value
        lower = np.percentile(valid_values, 0.1)  
        # Filter the dataframe to keep only values greater than or equal to this percentile
        df_filtered = df_filtered[df_filtered[metric] >= lower]
        print(f" - Trimmed {metric}: kept values >= {lower:.4f}")
    else:
        print(f" - No valid values for {metric}, skipping filtering.")

print(f"[INFO] Data shape after outlier removal: {df_filtered.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 4) Styled Violin Plot Loop (Integrated into a single Figure)
# ──────────────────────────────────────────────────────────────────────────────

# Create subplots with 1 row and 3 columns for the three metrics
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.2) # Adjust horizontal spacing between subplots

# Define the order of metrics to visualize
metric_cols = ["metaclip_score", "aesthetic_score", "perplexity_score"]

for i, metric_col in enumerate(metric_cols):
    cfg = metrics_to_plot[metric_col]
    metric_label = cfg["label"]
    # apply_log is removed from cfg
    ax = axes[i] # Assign the current subplot
    
    print(f"\n[PROCESS] Plotting {metric_label} on subplot {i+1}")
    
    # Use the filtered data for plotting
    df_plot = df_filtered.copy()
    
    # Assign the metric values to a generic column for plotting consistency
    df_plot["Metric_Value"] = df_plot[metric_col]
    
    # Drop any remaining NaN values before plotting
    df_plot = df_plot.dropna(subset=["Metric_Value", "prompt_length_group"])

    # Create a combination of violin plot and box plot
    sns.violinplot(
        x="prompt_length_group", y="Metric_Value",
        data=df_plot,
        order=bucket_order,
        palette=palette,
        linewidth=0.8,
        edgecolor='black',
        cut=1.5,
        ax=ax
    )

    sns.boxplot(
        x="prompt_length_group", y="Metric_Value",
        data=df_plot,
        order=bucket_order,
        width=0.15,
        palette=palette,
        showcaps=True, showfliers=False,
        boxprops   ={"facecolor": "white", "edgecolor": "black", "linewidth": 1.1},
        whiskerprops={"color": "black", "linewidth": 1.1, "linestyle": '-'},
        capprops   ={"color": "black", "linewidth": 1.1},
        medianprops={"color": "black", "linewidth": 1.6, "linestyle": '-'},
        ax=ax
    )

    # Style the axes
    ax.set_ylabel(metric_label, fontsize=20)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    
    # Add a black boundary to all 4 sides of the plot
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ax.spines['top'].set_edgecolor('black')
    ax.spines['right'].set_edgecolor('black')
    ax.spines['left'].set_edgecolor('black')
    ax.spines['bottom'].set_edgecolor('black')
    # Format the y-axis ticks to one decimal place
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.1f}'))

    # Remove the x-axis label from each subplot
    ax.set_xlabel("")

    # Adjust the y-axis limits to provide consistent padding
    ymin_data, ymax_data = df_plot["Metric_Value"].min(), df_plot["Metric_Value"].max()
    if np.isclose(ymin_data, ymax_data):
        lower_bound = ymin_data - 0.05
        upper_bound = ymax_data + 0.05
    else:
        range_of_data = ymax_data - ymin_data
        padding = range_of_data * 0.15
        lower_bound = ymin_data - padding
        upper_bound = ymax_data + padding
    
    # Ensure the lower bound is not negative
    lower_bound = max(lower_bound, 0.0) 

    ax.set_ylim(lower_bound, upper_bound)
    
# Automatically adjust subplot parameters for a tight layout
fig.tight_layout()

# Save the final figure
fname = "combined_violin_plots_horizontal.png"
fig.savefig(os.path.join(OUT_DIR, fname), dpi=600)
plt.close(fig)
print(f"\n[DONE] All plots combined and saved to → {fname}")