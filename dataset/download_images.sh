#!/bin/bash

# Define the script to run
PYTHON_SCRIPT="python3 ./diffusiondb/scripts/download.py"

# Define the arguments
START_INDEX=1
END_INDEX=2000
OUTPUT_DIR="./images/"

# --- Download Phase ---
echo "--- Downloading shards $START_INDEX to $END_INDEX ---"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the python script to download files
# The '-z' option ensures zip files are kept after download
$PYTHON_SCRIPT -i $START_INDEX -r $END_INDEX -o "$OUTPUT_DIR"

# --- Unzip and Cleanup Phase ---
echo "--- Unzipping files and cleaning up ---"

# Find all zip files in the output directory and unzip them
find "$OUTPUT_DIR" -type f -name "*.zip" | while read -r zipfile; do
    echo "Extracting $zipfile..."
    unzip -q "$zipfile" -d "$OUTPUT_DIR"
    if [ $? -eq 0 ]; then
        echo "Extraction successful. Deleting $zipfile."
        rm "$zipfile"
    else
        echo "Error extracting $zipfile. Keeping the file."
    fi
done

echo "--- All done! ---"