# 📄 README.md — Supplementary Code for Quantifying Human Contribution in Text-to-Image Generation through Comprehensive Analysis of Prompt Refinemen

> **Paper Title**: *“Quantifying Human Contribution in Text-to-Image Generation through Comprehensive Analysis of Prompt Refinement”*  
> **Track**: AI for Social Impact (AISI)  
> **Submitted to**: AAAI-26  

---

### 🌟 Project Overview
This repository contains the supplementary code for our research paper, which aims to quantify the human contribution in the iterative process of text-to-image generation through prompt refinement. 

### 📦 Repository Contents

```bash
.
├── README.md
├── dataset
│   ├── diffusiondb
│   ├── download_images.sh
│   ├── download_prompt.py
│   ├── images
│   └── manifest.txt
├── metrics
│   ├── bertscore.py
│   ├── aesthetic_score.py
│   ├── clip_score.py
│   └── ppl.py
├── filter.py
├── calculate_metrics.py
├── calculate_hc.py
└── requirement.txt
```

---
### 📌 Notes

- All models will automatically download from HuggingFace.
- Ensure GPU availability for large models (CLIP / Aesthetic / BERT / GPT-2).
- Code is modular: you may run each scoring independently.

### ⚙️ Setup Instructions

#### 1. Create Virtual Environment (Optional)

```bash
python -m venv hc_venv
source hc_venv/bin/activate  
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
#### 3. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

### 🚀 Workflow

The analysis pipeline involves several sequential steps, each handled by a dedicated script. The output of one script often serves as the input for the next.

1.  **Data Preparation (`dataset/download_prompt.py && dataset/download_images`)**

    * `dataset/download_images.sh`: This shell script is responsible for downloading the actual image files associated with the prompts from DiffusionDB into the `./dataset/images` directory. These images are necessary for calculating visual metrics like CLIP Score and Aesthetic Score.

    * `dataset/download_prompt.py`: This script downloads the raw metadata.parquet file from Hugging Face DiffusionDB. It then performs an initial filtering by dropping unnecessary columns and saves the cleaned metadata to `removed_metadata.parquet`.

2.  **Prompt Filtering and Categorization (`filter.py`)**

    * Reads the `removed_metadata.parquet` file.

    * **Filters out non-English or emoji-only prompts** using `EMOJI_PATTERN` and `is_english` functions.

    * **Extracts the main subject** from each prompt using spaCy.

    * **Normalizes subjects** based on predefined rules (e.g., mapping "fox anthro" to "anthropomorphic fox").

    * **Categorizes prompts into refinement threads** where a user repeatedly refines prompts about the same subject (at least 3 times).

    * **Analyzes prompt length trends** within these threads and saves only the "Increase_Length_Trend" data to `./dataset/prompt_thread/Increase_Length_Trend.csv`. This file is the primary input for subsequent metric calculations.

3.  **Metric Calculation (`calculate_metrics.py`)**

    * This is the core script for computing various quantitative metrics.

    * It reads the `Increase_Length_Trend.csv` file.

    * **PPL (Perplexity)**: Calculates the perplexity score for each prompt, indicating its linguistic fluency.

    * **BERTScore (Semantic Similarity)**: Computes the semantic similarity between consecutive prompts within each thread, measuring how much the prompt's meaning changes.

    * **CLIP Score**: Measures the visual-textual alignment between the generated image (referenced by `image_name`) and its corresponding prompt.

    * **Aesthetic Score**: Predicts the aesthetic quality of the generated image.

    * **Checkpointing**: Due to the time-consuming nature of these calculations, this script saves intermediate results to temporary files (`./tmp/`) after each metric is computed. This allows the process to resume from the last completed step if interrupted.

    * All calculated metrics are progressively added to a single DataFrame, which is finally saved to `./dataset/metrics.csv`.

4.  **Human Contribution Score Calculation (`calculate_hc.py`)**

    * Reads the `metrics.csv` file, which contains all the computed metrics.

    * Calculates `Q_k` (Image Quality) based on `metaclip_score` and `aesthetic_score`.

    * Determines `Q0` (initial image quality) and `delta Q_k` (change in image quality).

    * Calculates `M_k` (Modification Strength) using `perplexity reduction` and `semantic divergence`.

    * Finally, computes the **Human Contribution Score** by cumulatively summing `M_k * delta Q_k` and adding `Q0`.

    * The final results, including the Human Contribution Score, are saved to `./results/hc.csv`.


### ▶️ Usage

To run the full pipeline, execute the scripts in the following order from the project root directory:

1.  **Prepare Data:**

    ```
    # For images: Downloads image files into ./dataset/images
    # Ensure execute permissions: sudo chmod +x dataset/download_images.sh
    cd dataset
    git clone https://github.com/poloclub/diffusiondb
    bash dataset/download_images.sh

    # For prompts: Downloads and filters metadata, saving to removed_metadata.parquet
    python dataset/download_prompt.py
    ```

2.  **Filter and Categorize Prompts:**

    ```
    cd ../
    python filter.py
    ```

3.  **Calculate Metrics:**

    ```
    python calculate_metrics.py
    ```

4.  **Calculate Human Contribution Score:**

    ```
    python calculate_hc.py
    ```

The final Human Contribution Scores will be available in `./results/hc.csv`.

---

### 📄 License & Acknowledgements

- This code is provided as supplementary material for academic peer review only.
