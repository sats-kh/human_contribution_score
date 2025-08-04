import pandas as pd
import numpy as np
import os

def calculate_human_contribution_score(input_csv_path="./dataset/metrics.csv", output_dir="./results", output_filename="hc.csv"):
    """
    Calculates Human Contribution Score (PPL, BERTScore, CLIP, Aesthetic score)
    sequentially on the entire dataset and saves the result to a single CSV file.

    Args:
        input_csv_path (str): Path to the input CSV file containing all required metrics.
        output_dir (str): Directory to save the output file.
        output_filename (str): Name of the output CSV file.
    """

    # --- 1) Path Configuration ────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    print(f"Input file for human contribution score calculation: {input_csv_path}")

    # --- 2) Data Loading ─────────────────────────────────────────────
    try:
        df = pd.read_csv(input_csv_path)
        print(f"Successfully loaded {input_csv_path}")
        print("Initial DataFrame head:")
        print(df.head())
    except FileNotFoundError:
        print(f"Error: The file '{input_csv_path}' does not exist. Please check the file path.")
        return

    # Check for required columns
    required_columns = [
        'category_number',
        'metaclip_score',
        'aesthetic_score',
        'semantic_similarity',
        'perplexity_score',
        'prompt_sequence_number'
    ]

    if not set(required_columns).issubset(df.columns):
        missing_cols = list(set(required_columns) - set(df.columns))
        raise ValueError(f"The CSV file must contain all required columns: {missing_cols}")

    # Clean 'category_number': convert to integer and filter out rows where it's less than 1
    df['category_number'] = pd.to_numeric(df['category_number'], errors='coerce')
    df = df.dropna(subset=['category_number'])
    df['category_number'] = df['category_number'].astype(int)
    df = df[df['category_number'] >= 1].copy()

    # --- 4) Q_k (Image Quality) Calculation ──────────────────────────────────
    # Define alpha and beta values
    alpha = 0.73
    beta = 0.27

    df['Q_k'] = (alpha * df['metaclip_score']) + (beta * df['aesthetic_score'] * 10)

    print("\n--- 'Q_k' (Image Quality) calculation complete ---")
    print(df[['category_number', 'prompt_sequence_number', 'metaclip_score', 'aesthetic_score', 'Q_k']].head())

    # --- 6) Q0 and delta Q_k Calculation ───────────────────────────────────
    # Calculate the initial quality Q0 for each category
    df['Q0'] = df.groupby('category_number')['Q_k'].transform('first')

    # Calculate the previous Q_k value
    df['Q_k_prev'] = df.groupby('category_number')['Q_k'].shift(1)
    # Calculate the change in Q_k
    df['delta Q_k'] = df['Q_k'] - df['Q_k_prev']

    # Set negative delta Q_k values to 0
    df['delta Q_k'] = np.maximum(0, df['delta Q_k'])

    print("\n--- 'Q0' and 'delta Q_k' calculation complete ---")

    # --- 5) M_k (Modification Intensity) Calculation ───────────────────────────────────

    # 5-1) Calculate perplexity reduction
    df['perplexity_prev'] = df.groupby('category_number')['perplexity_score'].shift(1)
    
    # Calculate the reduction in perplexity.
    # This code calculates the simple difference as written in the user's latest provided script.
    df['perplexity_reduction'] = np.maximum(0, (df['perplexity_prev']) - (df['perplexity_score']))
    
    # 5-2) Calculate semantic divergence
    df['semantic_divergence'] = np.exp(1 - df['semantic_similarity'])
    # Set the first step's semantic divergence to NaN as there is no previous data
    df.loc[df['prompt_sequence_number'] == 1, 'semantic_divergence'] = np.nan

    # Calculate M_k
    df['M_k'] = df['semantic_divergence'] * df['perplexity_reduction']

    # Total Human Contribution (H-revised) Calculation (Applying cumulative summation logic)
    df['Mk_Qk'] = df['M_k'] * df['delta Q_k']
    df['Cumulative_Mk_Qk'] = df.groupby('category_number')['Mk_Qk'].cumsum()
    df['Human_Contribution_Score'] = df['Q0'] + df['Cumulative_Mk_Qk']

    # For the first prompt in each thread, the cumulative sum is NaN.
    # We set the score to Q0 in this case.
    df.loc[df['prompt_sequence_number'] == 1, 'Human_Contribution_Score'] = df['Q0']

    # Save the final DataFrame
    df.to_csv(output_path, index=False)
    print(f"\n--- Final results saved to '{output_path}'. ---")


if __name__ == '__main__':
    calculate_human_contribution_score()