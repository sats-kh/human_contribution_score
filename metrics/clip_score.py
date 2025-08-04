import pandas as pd
from PIL import Image
import os
import torch
from transformers import AutoProcessor, AutoModel
import numpy as np
import warnings

# This script calculates the CLIP score using a single model for a given CSV file.
# The scores are computed for each row and added as a new column.

# 1. Common file paths and settings
# Set the base directory for image files.
# This assumes images are in a directory relative to the script's location.
IMAGE_BASE_DIR = "./dataset/images"

def load_metaclip_model_and_processor(model_name="facebook/metaclip-h14-fullcc2.5b"):
    """Loads the specified MetaCLIP model and its corresponding processor."""
    print(f"Loading MetaCLIP model and processor: {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model.to('cuda')
        print(f"Model {model_name} moved to CUDA.")
    else:
        print(f"Model {model_name} loaded on CPU.")
    return processor, model

def calculate_metaclip_score(image_name, text_prompt, processor, model):
    """Calculates the MetaCLIP similarity score for a single image-text pair."""
    full_image_path = os.path.join(IMAGE_BASE_DIR, image_name)

    if not os.path.exists(full_image_path):
        return None

    try:
        image = Image.open(full_image_path).convert("RGB")
    except Exception as e:
        warnings.warn(f"Error opening image {full_image_path}: {e}. Skipping score calculation.")
        return None

    inputs = processor(text=text_prompt, images=image, return_tensors="pt", padding=True, truncation=True)
    
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    similarity = (image_features * text_features).sum(axis=-1).item()
    metaclip_score = 100.0 * similarity
    return metaclip_score

def calculate_metaclip_scores_for_csv(input_csv_file, output_csv_file, temp_output_csv_file, save_interval_rows=10000):
    """
    Reads a CSV file, calculates MetaCLIP scores for each row, and saves the results.
    Includes checkpointing to resume calculations if the process is interrupted.
    """
    print(f"\n--- Starting MetaCLIP score calculation for: {input_csv_file} ---")

    # Load the model and processor once
    processor, model = load_metaclip_model_and_processor()

    # --- Resume/Initial Load Logic ---
    start_index = 0
    df = None

    if os.path.exists(temp_output_csv_file):
        try:
            df_temp = pd.read_csv(temp_output_csv_file)
            last_processed_idx = df_temp['metaclip_score'].last_valid_index()
            if last_processed_idx is not None:
                start_index = last_processed_idx + 1
                df = df_temp
                print(f"Resuming from row {start_index} based on '{temp_output_csv_file}'.")
            else:
                df = pd.read_csv(input_csv_file)
                print(f"'{temp_output_csv_file}' found but no valid scores, starting from scratch.")
        except Exception as e:
            print(f"Error reading temporary file '{temp_output_csv_file}': {e}. Starting from scratch.")
            df = pd.read_csv(input_csv_file)
    else:
        df = pd.read_csv(input_csv_file)
        print(f"No temporary file found for '{input_csv_file}'. Starting from scratch.")
    # --- End Resume/Initial Load Logic ---

    print(f"Successfully loaded data. Total rows: {len(df)}")

    # Check for required columns
    if 'image_name' not in df.columns or 'prompt' not in df.columns:
        print("Error: 'image_name' or 'prompt' column not found in the CSV file.")
        print("Please ensure 'image_name' and 'prompt' columns exist.")
        return

    # Initialize score column if it doesn't exist
    if 'metaclip_score' not in df.columns:
        df['metaclip_score'] = np.nan

    print(f"\nSaving results to '{temp_output_csv_file}' every {save_interval_rows} rows.")

    total_rows = len(df)
    for index, row in df.iloc[start_index:].iterrows():
        image_name = row['image_name']
        prompt = row['prompt']
            
        if pd.isna(image_name) or pd.isna(prompt):
            df.at[index, 'metaclip_score'] = None
            continue
            
        if (index + 1) % 100 == 0:
            print(f"Processing row {index+1}/{total_rows}: Image='{image_name}'...")

        score = calculate_metaclip_score(image_name, prompt, processor, model)
        df.at[index, 'metaclip_score'] = score

        if (index + 1) % save_interval_rows == 0:
            df.to_csv(temp_output_csv_file, index=False)
            print(f"--- Intermediate results saved to '{temp_output_csv_file}' up to row {index} ---")

    # Final save and cleanup
    df.to_csv(output_csv_file, index=False)
    print(f"\nAll MetaCLIP scores calculated for '{input_csv_file}' and saved to '{output_csv_file}'.")

    if os.path.exists(temp_output_csv_file):
        os.remove(temp_output_csv_file)
        print(f"Temporary file '{temp_output_csv_file}' removed.")
