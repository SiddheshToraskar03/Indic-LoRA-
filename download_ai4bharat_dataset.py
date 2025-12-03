#!/usr/bin/env python3
"""
Download AI4Bharat dataset and save to local directory
This script downloads the dataset files directly without using deprecated scripts
"""

import os
import json
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import shutil

# Configuration
OUTPUT_DIR = "ai4bharat_dataset"
DATASET_NAME = "ai4bharat/indic_glue"
# Available subsets: 'wstp.hi', 'wstp.ta', 'wstp.te', 'wstp.ml', 'wstp.kn', 'wstp.gu', 'wstp.mr', 'wstp.bn', 'wstp.as', 'wstp.or', 'wstp.pa', etc.
# We'll try to download multiple language subsets and combine them
SUBSETS_TO_DOWNLOAD = [
    "wstp.hi",  # Hindi
    "wstp.ta",  # Tamil
    "wstp.te",  # Telugu
    "wstp.ml",  # Malayalam
    "wstp.kn",  # Kannada
    "wstp.gu",  # Gujarati
    "wstp.mr",  # Marathi
    "wstp.bn",  # Bengali
    "wstp.as",  # Assamese
    "wstp.or",  # Odia
    "wstp.pa",  # Punjabi
]

print("="*60)
print("Downloading AI4Bharat Dataset")
print("="*60)
print(f"Dataset: {DATASET_NAME}")
print(f"Subsets to download: {len(SUBSETS_TO_DOWNLOAD)} languages")
print(f"Output directory: {OUTPUT_DIR}")
print("="*60)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_data = []
successful_subsets = []
failed_subsets = []

for subset in SUBSETS_TO_DOWNLOAD:
    try:
        print(f"\n{'='*60}")
        print(f"Downloading subset: {subset}")
        print(f"{'='*60}")
        
        # Load the subset
        dataset = load_dataset(DATASET_NAME, subset, split="train")
        print(f"✓ Successfully loaded {subset}")
        print(f"  Number of examples: {len(dataset)}")
        print(f"  Features: {dataset.features}")
        
        # Add language info if not present
        for example in dataset:
            example_dict = dict(example)
            # Add language code if not present
            if 'language' not in example_dict and 'lang' not in example_dict:
                # Extract language from subset name (e.g., "wstp.hi" -> "hi")
                lang_code = subset.split('.')[-1] if '.' in subset else subset
                example_dict['lang_code'] = lang_code
            all_data.append(example_dict)
        
        successful_subsets.append(subset)
        print(f"  Added {len(dataset)} examples (Total so far: {len(all_data):,})")
        
    except Exception as e:
        failed_subsets.append((subset, str(e)))
        print(f"  ❌ Failed to load {subset}: {str(e)[:100]}...")
        continue

if not all_data:
    print("\n❌ Error: Could not download any subsets!")
    print("Failed subsets:")
    for subset, error in failed_subsets:
        print(f"  {subset}: {error[:100]}")
    raise RuntimeError("No data downloaded")

print(f"\n{'='*60}")
print("Saving combined dataset...")
print(f"{'='*60}")

# Save dataset as JSON
output_file = os.path.join(OUTPUT_DIR, "dataset.json")
print(f"Saving to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"✓ Saved {len(all_data):,} examples to {output_file}")

# Also save as JSONL (one example per line) for easier loading
output_jsonl = os.path.join(OUTPUT_DIR, "dataset.jsonl")
print(f"Saving to {output_jsonl}...")
with open(output_jsonl, 'w', encoding='utf-8') as f:
    for example in all_data:
        f.write(json.dumps(example, ensure_ascii=False) + '\n')

print(f"✓ Also saved as JSONL to {output_jsonl}")

# Save dataset info
info = {
    "dataset_name": DATASET_NAME,
    "subsets_downloaded": successful_subsets,
    "subsets_failed": [s[0] for s in failed_subsets],
    "num_examples": len(all_data),
    "features": str(dataset.features) if successful_subsets else "N/A"
}

with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), 'w', encoding='utf-8') as f:
    json.dump(info, f, indent=2)

print(f"\n✓ Dataset info saved to {OUTPUT_DIR}/dataset_info.json")

# Show summary
print("\n" + "="*60)
print("Download Summary:")
print("="*60)
print(f"Total examples: {len(all_data):,}")
print(f"Successful subsets: {len(successful_subsets)}")
print(f"Failed subsets: {len(failed_subsets)}")
if successful_subsets:
    print(f"\nSuccessfully downloaded:")
    for subset in successful_subsets:
        print(f"  ✓ {subset}")

# Show sample
if all_data:
    print("\n" + "="*60)
    print("Sample example:")
    print("="*60)
    print(json.dumps(all_data[0], indent=2, ensure_ascii=False))

print("\n" + "="*60)
print("✓ Download complete!")
print("="*60)
print(f"\nNext steps:")
print(f"1. Upload the '{OUTPUT_DIR}' folder to Google Drive")
print(f"2. Mount Google Drive in Colab")
print(f"3. Update the script to load from Drive path")

