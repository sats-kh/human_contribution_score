import pandas as pd
import numpy as np
import torch
from bert_score import score
from tqdm import tqdm
import os
import csv

def compute_bert_text_similarity(
    input_csv_path: str,
    output_csv_path: str = "bert_text_sim.csv",
    bert_model: str = "bert-large-uncased",
    batch_size: int = 1024,
    new_column_name: str = "semantic_similarity",
    device_override: str = None
):
    """
    Calculates BERTScore similarity between consecutive prompts within each category
    and saves the results to a new CSV file.

    Parameters:
    - input_csv_path: str – Path to the input CSV file.
    - output_csv_path: str – Path to save the CSV with the new BERTScore column.
    - bert_model: str – The BERT model name to use (default: 'bert-large-uncased').
    - batch_size: int – Batch size for BERTScore calculation (default: 1024).
    - new_column_name: str – Name of the new column to be added (default: 'semantic_similarity').
    - device_override: str – Optional override for the CUDA device (e.g., 'cuda:0').
    """

    # 1. Set device
    if device_override:
        device = device_override
    elif torch.cuda.is_available():
        device = "cuda"
        print("Calculating BERTScore using GPU (CUDA).")
    else:
        device = "cpu"
        print("Calculating BERTScore using CPU. This may take a while.")

    # 2. Load CSV
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"File '{input_csv_path}' not found.")

    df = pd.read_csv(input_csv_path)
    print(f"✅ File loaded successfully: {input_csv_path}")
    print(f"Total number of rows: {len(df)}")

    # 3. Drop nulls & sort
    essential_cols = ['category_number', 'timestamp', 'prompt']
    print("\nChecking for missing values:")
    print(df[essential_cols].isnull().sum())

    initial_len = len(df)
    df.dropna(subset=essential_cols, inplace=True)
    print(f"{initial_len - len(df)} missing rows removed.")
    df.sort_values(by=['category_number', 'timestamp'], inplace=True)

    # 4. Prepare BERTScore calculation
    df[new_column_name] = np.nan
    category_groups = df.groupby('category_number')

    print(f"\n🔍 Starting BERTScore calculation: model={bert_model}, device={device}")

    for category_num, group in tqdm(category_groups, desc="Calculating BERTScore by category", unit="category"):
        prompts = group['prompt'].tolist()
        if len(prompts) < 2:
            continue

        references = prompts[:-1]
        candidates = prompts[1:]
        category_scores = [np.nan]

        P, R, F1 = score(
            cands=candidates,
            refs=references,
            model_type=bert_model,
            lang='en',
            rescale_with_baseline=True,
            device=device,
            batch_size=batch_size,
            verbose=False
        )
        category_scores.extend(F1.tolist())
        df.loc[group.index, new_column_name] = category_scores

    print(f"\n✅ BERTScore calculation complete. New column: {new_column_name}")
    print(df[[ 'category_number', 'timestamp', 'prompt', new_column_name ]].head(10))

    # 5. Save CSV
    df.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_ALL, escapechar='\\')
    print(f"📁 Results saved to: {output_csv_path}")