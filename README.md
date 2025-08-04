# 📄 README.md — Supplementary Code for Quantifying Human Contribution in Text-to-Image Generation through Comprehensive Analysis of Prompt Refinemen

> **Paper Title**: *“Quantifying Human Contribution in Text-to-Image Generation through Comprehensive Analysis of Prompt Refinement”*  
> **Track**: AI for Social Impact (AISI)  
> **Submitted to**: AAAI-26  
<!-- > **Authors**: [Author List] -->

---

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

### ⚙️ Setup Instructions

#### 1. Create Virtual Environment (Optional)

```bash
python -m venv hc_venv
source hc_venv/bin/activate  # For Linux/Mac
```

#### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
---

### 🧪 Reproducing Metrics

#### 1. **BERTScore 계산**

```python
from bertscore_utils import compute_bert_text_similarity

compute_bert_text_similarity(
    input_csv_path="prompts.csv",
    output_csv_path="prompts_with_bertscore.csv"
)
```

#### 2. **CLIP Score + Aesthetic Score 계산**

```python
from clip_score_utils import (
    load_clip_model_and_processor,
    process_csv_for_clip_scores,
    add_aesthetic_scores_to_csv
)

# Load model
processor, model = load_clip_model_and_processor("facebook/metaclip-h14-fullcc2.5b")

# CLIP score
process_csv_for_clip_scores(
    input_csv_file="prompts.csv",
    output_csv_file="prompts_with_clip.csv",
    temp_output_csv_file="temp_clip.csv",
    processor=processor,
    model=model
)

# Aesthetic score
add_aesthetic_scores_to_csv(
    input_csv_path="prompts.csv",
    output_csv_path="prompts_with_aes.csv",
    image_root_dir="path/to/images"
)
```

#### 3. **Perplexity 계산**

```python
from clip_score_utils import add_perplexity_scores_to_csv

add_perplexity_scores_to_csv(
    input_csv_file="prompts.csv",
    output_csv_file="prompts_with_ppl.csv",
    temp_output_csv_file="temp_ppl.csv"
)
```

---

### 📁 Input Format

Your CSV file (e.g., `prompts.csv`) should contain at minimum the following columns:

| Column          | Description                         |
|------------------|-------------------------------------|
| `prompt`         | Text prompt to evaluate             |
| `image_name`     | (optional) Relative or absolute path to the image file |
| `timestamp`      | (optional) For BERTScore ordering   |
| `category_number`| (optional) Group ID for comparison  |

---

### 📌 Notes

- All models will automatically download from HuggingFace.
- Ensure GPU availability for large models (CLIP / Aesthetic / BERT / GPT-2).
- Code is modular: you may run each scoring independently.

---
<!-- 
### 📄 License & Acknowledgements

- This code is provided as supplementary material for academic peer review only. -->
