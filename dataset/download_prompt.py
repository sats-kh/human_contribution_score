from urllib.request import urlretrieve
import pandas as pd

# Download the metadata table in parquet format from Hugging Face
table_url = f'https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet'
urlretrieve(table_url, 'metadata.parquet')

# Read the downloaded parquet table into a Pandas DataFrame
metadata_df = pd.read_parquet('metadata.parquet')

# Print the original column headers of the raw dataset
print("Original Columns:")
print(metadata_df.columns)

# Define a list of columns to be removed, as they are not necessary for the current analysis
columns_to_drop = [
    'part_id',
    'seed',
    'step',
    'cfg',
    'sampler',
    'width',
    'height',
    'image_nsfw',
    'prompt_nsfw'
]

# Drop the specified columns from the DataFrame
metadata_df = metadata_df.drop(columns=columns_to_drop)

# Print the columns of the modified DataFrame to verify the changes
print("\nModified Columns:")
print(metadata_df.columns)

# Print the first 5 rows of the modified DataFrame for a quick look
print("\nFirst 5 rows of the modified DataFrame:")
print(metadata_df.head())

# Save the final DataFrame to a Parquet file
metadata_df.to_parquet('removed_model_parameters.parquet', index=False)