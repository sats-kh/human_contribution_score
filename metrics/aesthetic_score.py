import pandas as pd
from PIL import Image
import os
import torch
from tqdm import tqdm
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
import torch.nn as nn
import numpy as np
import urllib.request, ssl
import warnings

# Use the same IMAGE_BASE_DIR as clip_score.py
IMAGE_BASE_DIR = "./dataset/images"

@torch.no_grad()
def load_clip_and_aesthetic_head():
    """
    Loads the CLIP model and the pre-trained aesthetic predictor head.
    The aesthetic predictor head is downloaded from a public repository if it doesn't exist locally.
    """
    clip_id = "openai/clip-vit-large-patch14"
    print(f"Loading CLIP model and processor: {clip_id}...")
    proc = CLIPProcessor.from_pretrained(clip_id)
    clip = CLIPModel.from_pretrained(clip_id).eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip.to(device)

    # Define cache path and download the aesthetic head weights if necessary
    cache = Path.home() / ".cache/aesthetic_predictor"
    cache.mkdir(parents=True, exist_ok=True)
    wt = cache / "sa_0_4_vit_l_14_linear.pth"
    if not wt.exists():
        print(f"Downloading aesthetic predictor head weights to {wt}...")
        ssl._create_default_https_context = ssl._create_unverified_context
        url = ("https://github.com/LAION-AI/aesthetic-predictor/"
               "raw/main/sa_0_4_vit_l_14_linear.pth")
        urllib.request.urlretrieve(url, wt)
        print("Download complete.")

    # Load the aesthetic predictor head
    head = nn.Linear(768, 1)
    head.load_state_dict(torch.load(wt, map_location="cpu"))
    head.to(device).eval()
    print(f"Aesthetic predictor head loaded on {device}.")
    return proc, clip, head, device

@torch.no_grad()
def compute_aesthetic_scores_batch(img_paths, processor, clip_model, head, device):
    """
    Computes aesthetic scores for a batch of images using the CLIP model and a linear head.
    """
    imgs, valid_idx = [], []
    for i, p in enumerate(img_paths):
        try:
            # Check if the file path is valid
            if not os.path.exists(p) or not os.path.isfile(p):
                continue
            
            img = Image.open(p).convert("RGB")
            imgs.append(img)
            valid_idx.append(i)
        except Exception:
            warnings.warn(f"Error opening image '{p}'. Skipping.")
            continue

    if not imgs:
        return [None] * len(img_paths)

    # Process images and get CLIP features
    inputs = processor(images=imgs, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    feats = clip_model.get_image_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    
    # Compute scores using the aesthetic head
    scores = head(feats).squeeze(-1).cpu().tolist()

    # Map scores back to the original batch
    out = [None] * len(img_paths)
    for i, s in zip(valid_idx, scores):
        out[i] = s
    return out

def add_aesthetic_scores_to_csv(input_csv_path, output_csv_path=None, batch_size=64):
    """
    Calculates aesthetic scores for images listed in a CSV and adds them as a new column.
    
    Parameters:
    - input_csv_path: str – Path to the input CSV file. Must contain an 'image_name' column.
    - output_csv_path: str – Path to save the resulting CSV file.
    - batch_size: int – Batch size for calculating aesthetic scores.
    """
    csv_path = Path(input_csv_path).expanduser()
    out_csv = output_csv_path or csv_path.with_name(csv_path.stem + "_aes.csv")
    batch = max(1, batch_size)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file '{csv_path}' not found.")

    df = pd.read_csv(csv_path)
    if "image_name" not in df.columns:
        raise ValueError("The CSV file must contain an 'image_name' column.")
    
    # Initialize aesthetic score column with NaN
    if "aesthetic_score" not in df.columns:
        df["aesthetic_score"] = np.nan
    
    processor, clip_model, head, device = load_clip_and_aesthetic_head()
    
    # Check if a temporary file exists to resume calculations
    start_idx = 0
    temp_output_path = out_csv.with_name(f"temp_{out_csv.name}")
    if temp_output_path.exists():
        try:
            df_temp = pd.read_csv(temp_output_path)
            last_valid_idx = df_temp['aesthetic_score'].last_valid_index()
            if last_valid_idx is not None:
                start_idx = last_valid_idx + 1
                df.update(df_temp)
                print(f"Resuming calculation from index {start_idx} using '{temp_output_path}'.")
            else:
                print(f"Temporary file '{temp_output_path}' found but has no valid scores. Starting from scratch.")
        except Exception as e:
            print(f"Error reading temporary file '{temp_output_path}': {e}. Starting from scratch.")
    else:
        print("No temporary file found. Starting calculation from the beginning.")

    # Process data in batches
    for start in tqdm(range(start_idx, len(df), batch), unit="batch", desc="Computing aesthetic"):
        end = min(start + batch, len(df))
        paths = []
        for name in df["image_name"].iloc[start:end]:
            # Construct the full image path using the base directory
            full_path = os.path.join(IMAGE_BASE_DIR, str(name))
            paths.append(full_path)

        batch_scores = compute_aesthetic_scores_batch(paths, processor, clip_model, head, device)
        df["aesthetic_score"].iloc[start:end] = batch_scores

        # Save intermediate results to a temporary file
        df.to_csv(temp_output_path, index=False, encoding="utf-8")
    
    # Final save and cleanup
    df.to_csv(out_csv, index=False, encoding="utf-8")
    if temp_output_path.exists():
        os.remove(temp_output_path)
        print(f"\nTemporary file '{temp_output_path}' removed.")
    
    print(f"\n✓ Aesthetic scores saved to → {out_csv}")