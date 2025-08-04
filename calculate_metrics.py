import os
from metrics.ppl import add_perplexity_scores_to_csv
from metrics.bertscore import compute_bert_text_similarity
from metrics.clip_score import load_clip_model_and_processor, process_csv_for_clip_scores
from metrics.aesthetic_score import add_aesthetic_scores_to_csv

def main():
    """
    Calculates Human Contribution Metrics (PPL, BERTScore, CLIP, Aesthetic score)
    for the entire dataset and saves the results to a single CSV file.

    Note: The metrics are calculated and saved to separate, temporary CSV files
    in a step-by-step manner. This is because the calculations can be very time-consuming.
    Saving intermediate results acts as a checkpoint, allowing the process to
    resume from the last completed step if an error occurs.
    """
    
    # 1. Create Output Directories and Configure File Paths
    TEMP_DIR = "./tmp"
    OUTPUT_DIR = "./dataset"
    os.makedirs(TEMP_DIR, exist_ok=True)

    
    print("--- Human Contribution Metrics Calculation (FULL RUN) ---")
    
    # Original input file
    input_file = "./dataset/prompt_thread/Increase_Length_Trend.csv"

    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: The input file '{input_file}' was not found.")
        return
    
    # Configure all intermediate and final file paths
    final_output_file = os.path.join(OUTPUT_DIR, "metrics.csv")
    temp_ppl_file = os.path.join(TEMP_DIR, "temp_ppl_scores.csv")
    temp_clip_file = os.path.join(TEMP_DIR, "temp_clip_scores.csv")
    
    current_input_file = input_file
    
    # --- Step 1: Calculate Perplexity (PPL) ---
    print("\n[Step 1/4] Calculating Perplexity (PPL) scores...")
    try:
        ppl_output_file = os.path.join(TEMP_DIR, "ppl_scores.csv")
        add_perplexity_scores_to_csv(
            input_csv_file=current_input_file,
            output_csv_file=ppl_output_file,
            temp_output_csv_file=temp_ppl_file,
            model_name="gpt2",
            batch_size=32,
            save_interval_batches=100
        )
        current_input_file = ppl_output_file
        print("✔ PPL calculation step completed.")
    except Exception as e:
        print(f"\nAn error occurred during PPL calculation: {e}")
        return

    # --- Step 2: Calculate BERTScore Text Similarity ---
    print("\n[Step 2/4] Calculating BERTScore text similarity...")
    try:
        bert_output_file = os.path.join(TEMP_DIR, "ppl_bert_scores.csv")
        compute_bert_text_similarity(
            input_csv_path=current_input_file,
            output_csv_path=bert_output_file,
            bert_model="bert-large-uncased",
            batch_size=1024,
            new_column_name="semantic_similarity"
        )
        current_input_file = bert_output_file
        print("✔ BERTScore calculation step completed.")
    except Exception as e:
        print(f"\nAn error occurred during BERTScore calculation: {e}")
        return

    # --- Step 3: Calculate CLIP Score ---
    print("\n[Step 3/4] Calculating CLIP scores...")
    try:
        clip_processor, clip_model = load_clip_model_and_processor(
            "openai/clip-vit-large-patch14"
        )
        clip_output_file = os.path.join(TEMP_DIR, "ppl_bert_clip_scores.csv")
        process_csv_for_clip_scores(
            input_csv_file=current_input_file,
            output_csv_file=clip_output_file,
            temp_output_csv_file=temp_clip_file,
            processor=clip_processor,
            model=clip_model
        )
        current_input_file = clip_output_file
        print("✔ CLIP score calculation step completed.")
    except Exception as e:
        print(f"\nAn error occurred during CLIP score calculation: {e}")
        return
        
    # --- Step 4: Calculate Aesthetic Score ---
    print("\n[Step 4/4] Calculating Aesthetic scores...")
    try:
        add_aesthetic_scores_to_csv(
            input_csv_path=current_input_file,
            output_csv_path=final_output_file
        )
        current_input_file = final_output_file
        print("✔ Aesthetic score calculation step completed.")
    except Exception as e:
        print(f"\nAn error occurred during Aesthetic score calculation: {e}")
        return
        
    print("\n--- All metrics calculation completed successfully. ---")
    print(f"Final results are saved to '{final_output_file}'.")
    
    # Clean up temporary files
    try:
        os.remove(os.path.join(TEMP_DIR, "ppl_scores.csv"))
        os.remove(os.path.join(TEMP_DIR, "ppl_bert_scores.csv"))
        os.remove(os.path.join(TEMP_DIR, "ppl_bert_clip_scores.csv"))
        if os.path.exists(os.path.join(TEMP_DIR, "temp_ppl_scores.csv")):
            os.remove(os.path.join(TEMP_DIR, "temp_ppl_scores.csv"))
        if os.path.exists(os.path.join(TEMP_DIR, "temp_clip_scores.csv")):
            os.remove(os.path.join(TEMP_DIR, "temp_clip_scores.csv"))
        print("\n✔ Intermediate files cleaned up.")
    except OSError:
        pass

if __name__ == '__main__':
    main()