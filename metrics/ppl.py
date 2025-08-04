import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import os

def load_lm_model_and_tokenizer(model_name="gpt2"):
    """
    Loads a pre-trained language model and its tokenizer, and handles device placement.
    """
    print(f"Loading Language Model and Tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Add a pad token if the tokenizer does not have one
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained(model_name)
    if torch.cuda.is_available():
        model.to('cuda')
        print(f"Model {model_name} moved to CUDA.")
    else:
        print(f"Model {model_name} loaded on CPU.")
    # Resize model token embeddings to match the new tokenizer size
    model.resize_token_embeddings(len(tokenizer))
    return tokenizer, model

def calculate_perplexity_batch(texts, tokenizer, model, device, max_length=512):
    """
    Calculates perplexity for a batch of text prompts.
    Returns a list of perplexity scores, with NaN for any invalid inputs.
    """
    if not texts:
        return []
    
    # Filter out invalid texts and keep track of original indices
    valid_texts_with_indices = [(i, text) for i, text in enumerate(texts) if pd.notna(text) and isinstance(text, str)]
    if not valid_texts_with_indices:
        return [np.nan] * len(texts)

    original_indices = [idx for idx, _ in valid_texts_with_indices]
    filtered_texts = [text for _, text in valid_texts_with_indices]
    
    # Tokenize the filtered batch, handling padding and truncation
    inputs = tokenizer(filtered_texts, return_tensors="pt", padding="longest", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    # Calculate token-level and sequence-level loss
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = inputs["input_ids"][..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
    loss_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss_per_sequence = loss_per_token.view(shift_labels.size()).sum(1)
    actual_token_counts = (shift_labels != tokenizer.pad_token_id).sum(1)

    # Calculate perplexity for each valid sequence
    ppls_valid = []
    for i in range(len(loss_per_sequence)):
        if actual_token_counts[i] > 0:
            avg_loss = loss_per_sequence[i] / actual_token_counts[i]
            ppl_score = torch.exp(avg_loss).item()
            ppls_valid.append(np.log(ppl_score)) # Using log-perplexity
        else:
            ppls_valid.append(np.nan)

    # Map the calculated scores back to the original batch size
    full_batch_ppls = [np.nan] * len(texts)
    for i, original_idx in enumerate(original_indices):
        full_batch_ppls[original_idx] = ppls_valid[i]

    return full_batch_ppls

def add_perplexity_scores_to_csv(input_csv_file, output_csv_file, temp_output_csv_file, model_name="gpt2",
                                  batch_size=32, save_interval_batches=100):
    """
    Main function to read a CSV, calculate perplexity for prompts in batches,
    and save the results with checkpointing to a new CSV file.
    """
    tokenizer, model = load_lm_model_and_tokenizer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_index = 0
    df = None
    
    # Check for a temporary file to resume calculation
    if os.path.exists(temp_output_csv_file):
        try:
            df_temp = pd.read_csv(temp_output_csv_file)
            last_processed_idx = df_temp['perplexity_score'].last_valid_index()
            if last_processed_idx is not None:
                start_index = last_processed_idx + 1
                df = df_temp.copy()
                print(f"Resuming from row {start_index} based on '{temp_output_csv_file}'.")
            else:
                df = pd.read_csv(input_csv_file)
                print(f"'{temp_output_csv_file}' found but no valid 'perplexity_score', starting from scratch.")
        except Exception as e:
            print(f"Error reading temporary file '{temp_output_csv_file}': {e}. Starting from scratch.")
            df = pd.read_csv(input_csv_file)
    else:
        df = pd.read_csv(input_csv_file)
        print(f"No temporary file found. Starting from scratch.")

    print(f"Successfully loaded data. Total rows: {len(df)}")

    # Ensure the 'prompt' column exists
    if 'prompt' not in df.columns:
        print("Error: 'prompt' column not found in the CSV file. Exiting.")
        return

    # Initialize the 'perplexity_score' column if it doesn't exist
    if 'perplexity_score' not in df.columns:
        df['perplexity_score'] = np.nan

    print(f"\nCalculating Perplexity with batch size: {batch_size}, saving every {save_interval_batches} batches.")
    total_rows = len(df)

    # Iterate through the data in batches, calculate scores, and save intermediate results
    for i in tqdm(range(start_index, total_rows, batch_size), desc="Calculating PPL"):
        batch_df = df.iloc[i : i + batch_size]
        # Skip batch if scores are already calculated
        if batch_df['perplexity_score'].notna().all():
            continue
        
        prompts_batch = batch_df['prompt'].tolist()
        original_global_indices = batch_df.index.tolist()
        
        ppl_scores = calculate_perplexity_batch(prompts_batch, tokenizer, model, device)
        
        for k, score in enumerate(ppl_scores):
            if pd.notna(score):
                df.at[original_global_indices[k], 'perplexity_score'] = score

        # Save intermediate results to the temporary file
        if (i // batch_size + 1) % save_interval_batches == 0 or (i + batch_size >= total_rows):
            df.to_csv(temp_output_csv_file, index=False)
            print(f"\n--- Intermediate results saved to '{temp_output_csv_file}' up to row {min(i + batch_size, total_rows) - 1} ---")

    # Save the final results and clean up the temporary file
    df.to_csv(output_csv_file, index=False)
    if os.path.exists(temp_output_csv_file):
        os.remove(temp_output_csv_file)
        print(f"Temporary file '{temp_output_csv_file}' removed.")

    print(f"\nPerplexity scores calculated and saved to '{output_csv_file}'.")