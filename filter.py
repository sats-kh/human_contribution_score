import pandas as pd
import spacy
import numpy as np
import os
import re
from tqdm.auto import tqdm
from transformers import CLIPTokenizer
from scipy.stats import linregress

# Configure tqdm to work with pandas apply functions
tqdm.pandas()

# --- 1. Filtering functions and global variables ──────────────────────────────
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

def is_english(text: str) -> bool:
    """Checks if the text is predominantly composed of English characters."""
    if not text or not isinstance(text, str):
        return False
    
    num_english_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')
    num_total_chars = len(text)
    
    if num_total_chars < 5:
        return num_english_chars / (num_total_chars + 1e-9) > 0.8
        
    return num_english_chars / num_total_chars > 0.5

def filter_non_english_and_emoji(df: pd.DataFrame, text_column: str = 'prompt') -> pd.DataFrame:
    """Filters out emoji-only or non-English prompts."""
    print("\n--- Starting Emoji/Non-English Prompt Filtering ---")
    initial_rows = len(df)
    
    df = df.dropna(subset=[text_column])
    df = df[df[text_column].apply(lambda x: isinstance(x, str))]

    df = df[~df[text_column].str.match(EMOJI_PATTERN) & df[text_column].apply(is_english)].copy()
    
    print(f"Rows remaining after filtering: {len(df)}")
    print(f"Rows removed: {initial_rows - len(df)}")
    return df

# --- 2. Subject extraction and normalization functions ──────────────────────────
def extract_main_subject_improved_from_doc(doc) -> str:
    """Extracts the main subject from a spaCy doc object."""
    if not doc or not doc.text.strip():
        return np.nan

    for token in doc:
        if token.lower_ == "of" and token.head.pos_ in {"NOUN", "PROPN", "VERB"}:
            for child in token.children:
                if child.dep_ == "pobj":
                    return child.text
    
    for tok in doc:
        if tok.dep_ in {"nsubj", "dobj", "pobj"} and tok.pos_ in {"NOUN", "PROPN"}:
            for chunk in doc.noun_chunks:
                if tok.i >= chunk.start and tok.i < chunk.end:
                    return chunk.text

    return next((chunk.text for chunk in doc.noun_chunks), np.nan)

def normalize_subject_rule_based(subject, subject_mapping_rules):
    """Standardizes the subject based on predefined rules."""
    subject_lower = str(subject).lower()
    for standard_name, keywords in subject_mapping_rules.items():
        for keyword in keywords:
            if keyword in subject_lower:
                return standard_name
    return subject_lower

# --- 3. CLIP token length filtering function ─────────────────────────────────
def calculate_and_filter_by_clip_token_length(df: pd.DataFrame, prompt_column: str = 'prompt', max_tokens: int = 77, model_name: str = "openai/clip-vit-large-patch14"):
    """
    Calculates prompt length using a CLIP tokenizer and removes data exceeding the max token length.
    """
    print("\n--- Starting CLIP Token Length Filtering ---")
    initial_rows = len(df)
    
    try:
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
    except OSError:
        print(f"Error: CLIP model '{model_name}' not found. Please check the model name.")
        return df

    df = df.dropna(subset=[prompt_column]).copy()
    
    print(f"CLIP tokenizer loaded. Max allowed tokens: {max_tokens}")
    print("Calculating actual CLIP token lengths for each prompt...")
    
    tqdm.pandas(desc="Calculating CLIP token lengths")
    df['actual_clip_token_length'] = df[prompt_column].progress_apply(
        lambda x: len(tokenizer.encode(str(x), truncation=False))
    )

    df_filtered = df[df['actual_clip_token_length'] <= max_tokens].copy()
    
    filtered_out_count = initial_rows - len(df_filtered)
    print(f"Total number of prompts: {initial_rows}")
    print(f"Prompts removed for exceeding {max_tokens} tokens: {filtered_out_count}")
    print(f"Prompts remaining after filtering: {len(df_filtered)}")
    
    return df_filtered

# --- 4. Data loading and initial filtering function ────────────────────────────
def load_and_initial_filter_data(file_name: str, prompt_column: str, user_id_column: str) -> pd.DataFrame:
    """Loads and performs initial filtering on the data."""
    print("--- Starting Data Loading and Initial Filtering ---")
    
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Error: The file '{file_name}' does not exist. Please check the file path.")

    metadata_df = pd.read_parquet(file_name)
    print(f"Original DataFrame size: {metadata_df.shape}")

    df_unique_prompts = metadata_df.drop_duplicates(subset=[prompt_column], keep='first')
    print(f"DataFrame size after removing duplicate prompts: {df_unique_prompts.shape}")

    if user_id_column in df_unique_prompts.columns:
        user_prompt_counts = df_unique_prompts[user_id_column].value_counts()
        users_with_3_or_more_prompts = user_prompt_counts[user_prompt_counts >= 3].index.tolist()
        df_filtered_initial = df_unique_prompts[df_unique_prompts[user_id_column].isin(users_with_3_or_more_prompts)].copy()
        print(f"Final filtered DataFrame size (users with < 3 prompts removed): {df_filtered_initial.shape}")
    else:
        print(f"Warning: The column '{user_id_column}' does not exist, so user-based filtering cannot be performed.")
        df_filtered_initial = df_unique_prompts.copy()
        
    return df_filtered_initial

# --- 5. Subject extraction and categorization function ──────────────────────────
def process_and_categorize_subjects(df: pd.DataFrame, user_id_column: str) -> pd.DataFrame:
    """
    Extracts subjects using spaCy, groups them based on rules, and assigns category numbers.
    """
    print("\n--- Starting prompt_subject extraction with spaCy ---")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("spaCy 'en_core_web_sm' model loaded successfully.")
    except OSError:
        raise OSError("The spaCy 'en_core_web_sm' model was not found. Please download it with 'python -m spacy download en_core_web_sm'.")

    df['prompt'] = df['prompt'].fillna('').astype(str)
    processed_docs = tqdm(nlp.pipe(df["prompt"], batch_size=2000, n_process=-1), 
                          total=len(df), desc="Processing prompts with spaCy")
    df["prompt_subject"] = [extract_main_subject_improved_from_doc(doc) for doc in processed_docs]
    print("prompt_subject extraction complete!")

    print("\n--- Starting Subject Normalization and Category Numbering ---")
    df.dropna(subset=['prompt_subject'], inplace=True)
    df['prompt_subject'] = df['prompt_subject'].astype(str).str.lower().str.strip()
    df = df[df['prompt_subject'].str.len() > 0]
    print(f"Keeping {len(df)} prompts after removing NaN and empty prompt_subject values.")

    subject_mapping_rules = {
        'anthropomorphic fox': ['anthropomorphic fox', 'fox anthro'],
        'anthropomorphic wolf': ['anthropomorphic wolf', 'wolf anthro'],
        'fursuit': ['fursuit', 'fursuiter'],
        'dragon': ['dragon', 'wyvern', 'drake'],
        'cat': ['cat', 'kitty', 'kitten'],
        'dog': ['dog', 'puppy'],
        'horse': ['horse', 'pony', 'stallion'],
        'bird': ['bird', 'robin', 'sparrow', 'finch'],
        'wolf': ['wolf', 'werewolf'],
        'superhero': ['superhero', 'supervillain', 'hero', 'villain'],
        'elf': ['elf', 'elvish'],
        'knight': ['knight', 'paladin', 'crusader'],
        'robot': ['robot', 'cyborg', 'android', 'mecha'],
        'samurai': ['samurai', 'ninja', 'shogun'],
        'wizard': ['wizard', 'sorcerer', 'warlock', 'mage', 'witch'],
        'beer bottle': ['beer bottle', 'bottle of beer', 'bottle'],
        'motorcycle': ['motorcycle', 'motorbike', 'scooter'],
        'castle': ['castle', 'fortress', 'citadel'],
        'spaceship': ['spaceship', 'starship', 'ufo'],
        'cityscape': ['cityscape', 'metropolis', 'urban landscape', 'city'],
        'tree': ['tree', 'oak', 'maple', 'pine', 'birch'],
        'car': ['car', 'automobile', 'vehicle'],
        'steampunk': ['steampunk', 'clockwork'],
        'cyberpunk': ['cyberpunk', 'futuristic city', 'neon city'],
        'fantasy': ['fantasy', 'magic kingdom', 'enchanted forest'],
        'gothic': ['gothic', 'gothic architecture', 'gothic cathedral'],
        'anime': ['anime', 'manga', 'kawaii']
    }
    df['normalized_subject'] = df['prompt_subject'].apply(lambda x: normalize_subject_rule_based(x, subject_mapping_rules))
    print(f"Rule-based Subject normalization complete. Number of unique normalized_subject: {df['normalized_subject'].nunique()}")

    category_counter = 0
    df['category_number'] = -1
    grouped_by_user = df.groupby(user_id_column)
    filtered_data_rows = []

    for user_name, user_df in tqdm(grouped_by_user, desc="Processing users for categorization"):
        user_df = user_df.sort_index()
        current_subject = None
        current_sequence_count = 0
        start_index_of_sequence = -1

        for idx, row in user_df.iterrows():
            subject = row['normalized_subject']
            if subject == current_subject:
                current_sequence_count += 1
            else:
                if current_sequence_count >= 3:
                    for prev_idx in range(start_index_of_sequence, idx):
                        if prev_idx in user_df.index:
                            user_row = user_df.loc[prev_idx].copy()
                            user_row['category_number'] = category_counter
                            filtered_data_rows.append(user_row)
                    category_counter += 1
                current_subject = subject
                current_sequence_count = 1
                start_index_of_sequence = idx
        
        if current_sequence_count >= 3:
            for prev_idx in range(start_index_of_sequence, user_df.index[-1] + 1):
                if prev_idx in user_df.index:
                    user_row = user_df.loc[prev_idx].copy()
                    user_row['category_number'] = category_counter
                    filtered_data_rows.append(user_row)
            category_counter += 1

    if filtered_data_rows:
        return pd.DataFrame(filtered_data_rows).drop_duplicates().sort_values(by=[user_id_column, 'category_number', 'prompt']).reset_index(drop=True)
    else:
        return pd.DataFrame(columns=df.columns.tolist() + ['normalized_subject', 'category_number'])

# --- 6. Prompt length trend analysis and saving function ──────────────────────
def analyze_prompt_length_trend(df: pd.DataFrame, output_dir: str):
    """Analyzes prompt length trends and saves the data by group."""
    print("\n--- Starting Prompt Length Trend Analysis ---")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Create 'prompt_length' column (word count)
    if 'prompt_length' not in df.columns:
        print("\n--- 'prompt_length' column not found. Calculating word count from 'prompt' column. ---")
        df['prompt_length'] = df['prompt'].apply(lambda x: len(str(x).split()))
        print("`prompt_length` column created.")

    # 2. Clean 'category_number' and sort
    df['category_number'] = pd.to_numeric(df['category_number'], errors='coerce', downcast='integer')
    df = df.dropna(subset=['category_number'])
    df['category_number'] = df['category_number'].astype(int)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['category_number', 'timestamp']).reset_index(drop=True)
    else:
        df = df.sort_values(by=['category_number']).reset_index(drop=True)
    
    df['prompt_sequence_number'] = df.groupby('category_number').cumcount() + 1
    print("\n--- 'prompt_sequence_number' column created ---")

    # 3. Calculate length change slope for each session
    def calculate_length_slope(group):
        if len(group) < 2:
            return np.nan 
        slope, _, _, _, _ = linregress(group['prompt_sequence_number'], group['prompt_length'])
        return slope

    print("\n--- Calculating prompt length change slope for each session ---")
    session_slopes = df.groupby('category_number').apply(calculate_length_slope).rename('length_slope').reset_index()
    df = pd.merge(df, session_slopes, on='category_number', how='left')
    print("`length_slope` column added.")

    # 4. Classify and save groups
    slope_threshold = 1.0
    def assign_length_group(slope):
        if pd.isna(slope):
            return 'Maintain_Single_Prompt_Session'
        elif slope > slope_threshold:
            return 'Increase_Length_Trend'
        elif slope < -slope_threshold:
            return 'Decrease_Length_Trend'
        else:
            return 'Maintain_Length_Trend'

    df['prompt_length_trend_group'] = df['length_slope'].apply(assign_length_group)
    print("\n--- Prompt length trend groups assigned ---")
    print("Data count for each group:")
    print(df['prompt_length_trend_group'].value_counts())

    group_names = df['prompt_length_trend_group'].unique()
    for group_name in group_names:
        group_df = df[df['prompt_length_trend_group'] == group_name].copy()
        sanitized_group_name = group_name.replace(" ", "_").replace("/", "_").replace("-", "_")
        output_filename = os.path.join(output_dir, f"{sanitized_group_name}.csv")
        group_df.to_csv(output_filename, index=False)
        print(f"✔ Saved '{group_name}' group data to '{output_filename}'. (Row count: {len(group_df)})")

    print("\nAll group classification and saving are complete.")
    return df

# --- 7. Main function ────────────────────────────────────────────────
def main():
    """Main function to run the entire data processing workflow."""
    file_name = './dataset/removed_model_parameters.parquet'
    prompt_column = 'prompt'
    user_id_column = 'user_name'
    output_final_csv_name = 'processed_categorized_prompts.csv'
    output_thread_dir = './dataset/prompt_thread'

    try:
        # 1. Load and perform initial filtering on the data
        df = load_and_initial_filter_data(file_name, prompt_column, user_id_column)
        
        # 2. Filter out emoji and non-English prompts
        df = filter_non_english_and_emoji(df, text_column=prompt_column)
        
        # 3. Filter by CLIP token length
        df = calculate_and_filter_by_clip_token_length(df, prompt_column, max_tokens=77)
        
        # 4. Extract subjects and categorize
        filtered_df = process_and_categorize_subjects(df, user_id_column)
                
        # 5. Analyze prompt length trends and save by group
        if not filtered_df.empty:
            final_df = analyze_prompt_length_trend(filtered_df, output_thread_dir)
            
            # Save the final combined result to a single CSV (if needed)
            final_df.to_csv(output_final_csv_name, index=False, encoding='utf-8-sig')
            print(f"\nAll processing is complete, and the final data has been saved to '{output_final_csv_name}'.")
        else:
            print("\nNo data remains after filtering, so final processing cannot be performed.")

    except FileNotFoundError as e:
        print(e)
    except KeyError as e:
        print(f"Error: Required column(s) are missing from the DataFrame. Missing column: {e}")
        print("Please check that the column names are 'prompt', 'user_name', etc.")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")

if __name__ == '__main__':
    main()