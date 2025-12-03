#!/usr/bin/env python3
"""
🦥 FineTuning With Unsloth - Multilingual Indian Languages

This script finetunes the Llama-3.2-3B-Instruct model using unsloth
on a multilingual dataset containing all Indian languages
(Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati,
Marathi, Punjabi, Odia, Assamese, Urdu, and more) to empower
the model with multilingual capabilities.

SETUP INSTRUCTIONS FOR GOOGLE COLAB:
1. Run download_ai4bharat_dataset.py locally to download the dataset
2. Upload the 'ai4bharat_dataset' folder to your Google Drive
3. In Colab, mount Google Drive by running:
   from google.colab import drive
   drive.mount('/content/drive')
4. Set LOAD_FROM_DRIVE = True below and update DRIVE_PATH to your Drive path
"""

# Install dependencies (run these commands manually or in your environment)
# !pip install -q unsloth
# !pip install -q --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git

import torch
import sys
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# ============================================================================
# Model Setup
# ============================================================================
max_seq_length = 2048  # Choose any! Unsloth also supports RoPE scaling internally.
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True  # Use 4bit quantization to reduce memory usage

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# ============================================================================
# Load Multilingual Indian Languages Dataset
# ============================================================================
# Configuration: Set to True if loading from Google Drive in Colab
LOAD_FROM_DRIVE = True  # Set to True when using Google Drive
# Common Drive paths - the script will try these automatically
POSSIBLE_DRIVE_PATHS = [
    "/content/drive/MyDrive/ai4bharat_dataset",
    "/content/drive/My Drive/ai4bharat_dataset",  # Note: "My Drive" with space
    "/content/drive/MyDrive/ai4bharat_dataset/dataset.jsonl",
    "/content/drive/My Drive/ai4bharat_dataset/dataset.jsonl",
]

# Load multilingual Indian languages dataset from AI4Bharat
# Option 1: Load from Google Drive (if you uploaded the dataset there)
# Option 2: Try loading directly from HuggingFace (may fail due to deprecated scripts)

# Try loading dataset - first from Drive, then from HuggingFace
dataset_loaded = False
last_error = None

# Option 1: Load from Google Drive (if uploaded)
if LOAD_FROM_DRIVE:
    import os
    print("Attempting to load from Google Drive...")
    
    # Try different possible paths
    found_path = None
    for path in POSSIBLE_DRIVE_PATHS:
        # Check if it's a direct file path
        if path.endswith('.jsonl') or path.endswith('.json'):
            if os.path.exists(path):
                found_path = path
                print(f"  Found dataset file: {path}")
                break
        # Check if it's a directory
        elif os.path.isdir(path):
            jsonl_path = os.path.join(path, "dataset.jsonl")
            json_path = os.path.join(path, "dataset.json")
            if os.path.exists(jsonl_path):
                found_path = jsonl_path
                print(f"  Found dataset.jsonl in: {path}")
                break
            elif os.path.exists(json_path):
                found_path = json_path
                print(f"  Found dataset.json in: {path}")
                break
    
    if found_path:
        try:
            print(f"  Loading from: {found_path}")
            dataset = load_dataset("json", data_files=found_path, split="train")
            print("✓ Successfully loaded from Google Drive")
            dataset_loaded = True
        except Exception as e:
            last_error = e
            print(f"  Failed to load from Drive: {str(e)[:150]}...")
            print("  Falling back to HuggingFace...")
    else:
        print("  ❌ Could not find dataset in any of these locations:")
        for path in POSSIBLE_DRIVE_PATHS:
            exists = os.path.exists(path) if not path.endswith(('.jsonl', '.json')) else os.path.exists(path)
            print(f"    - {path} {'(exists)' if exists else '(not found)'}")
        
        # Try to search for the file
        print("\n  🔍 Searching for dataset files in Drive...")
        try:
            import glob
            search_paths = [
                "/content/drive/MyDrive/**/dataset.jsonl",
                "/content/drive/My Drive/**/dataset.jsonl",
                "/content/drive/MyDrive/**/dataset.json",
                "/content/drive/My Drive/**/dataset.json",
            ]
            for pattern in search_paths:
                matches = glob.glob(pattern, recursive=True)
                if matches:
                    print(f"  ✓ Found: {matches[0]}")
                    found_path = matches[0]
                    dataset = load_dataset("json", data_files=found_path, split="train")
                    print("✓ Successfully loaded from Google Drive (auto-detected)")
                    dataset_loaded = True
                    break
        except Exception as search_error:
            print(f"  Search failed: {str(search_error)[:100]}...")
        
        if not dataset_loaded:
            print("\n  💡 To find your Drive path:")
            print("    1. In Colab, run: !ls /content/drive/MyDrive/")
            print("    2. Or check: !ls '/content/drive/My Drive/'")
            print("    3. Update POSSIBLE_DRIVE_PATHS with the correct path")
            print("  Falling back to HuggingFace...")

# Option 2: Try loading from HuggingFace (may fail due to deprecated scripts)
if not dataset_loaded:
    print("Attempting to load from HuggingFace...")
    
    # Try valid subsets that we know exist
    valid_subsets = ["wstp.hi", "wstp.ta", "wstp.te", "wstp.ml", "wstp.kn", "wstp.gu", "wstp.mr", "wstp.bn"]
    
    for subset in valid_subsets:
        try:
            print(f"Trying ai4bharat/indic_glue ({subset})...")
            dataset = load_dataset("ai4bharat/indic_glue", subset, split="train")
            print(f"✓ Successfully loaded ai4bharat/indic_glue ({subset})")
            print(f"  Note: This is only one language. For multilingual, use Drive option.")
            dataset_loaded = True
            break
        except Exception as e:
            last_error = e
            print(f"  Failed: {str(e)[:100]}...")
            continue

if not dataset_loaded:
    print("\n❌ Error: Could not load dataset from Drive or HuggingFace.")
    print(f"\nLast error: {last_error}")
    print("\nTo fix this:")
    print("1. Run download_ai4bharat_dataset.py locally to download the dataset")
    print("2. Upload the 'ai4bharat_dataset' folder to Google Drive")
    print("3. Mount Google Drive in Colab")
    print("4. Set LOAD_FROM_DRIVE = True and update DRIVE_PATH in this script")
    print("5. Or check: https://huggingface.co/datasets?search=ai4bharat for working datasets")
    raise RuntimeError("Could not load any dataset. Please follow the instructions above.")

# If your dataset has multiple splits, you might want to combine them:
# dataset = load_dataset("dataset_name", split="train+validation")

# Inspect the dataset structure
print("\n" + "="*60)
print("Dataset Info:")
print("="*60)
print(f"Dataset: {dataset}")
print(f"Features: {dataset.features}")
print(f"Number of examples: {len(dataset)}")
print("\nFirst 3 examples:")
for i in range(min(3, len(dataset))):
    print(f"\nExample {i+1}:")
    print(dataset[i])
print("="*60)

# Analyze and sample language distribution
print("\n" + "="*60)
print("Language Distribution Analysis:")
print("="*60)

from collections import Counter, defaultdict
import random

# Set random seed for reproducible sampling
random.seed(3407)  # Same seed as training for consistency

# Configuration: Maximum samples per language
MAX_SAMPLES_PER_LANGUAGE = 30000

try:
    # Check if dataset has a language field
    lang_field = None
    if 'language' in dataset.features:
        lang_field = 'language'
    elif 'lang' in dataset.features:
        lang_field = 'lang'
    elif 'lang_code' in dataset.features:
        lang_field = 'lang_code'
    elif 'lang_id' in dataset.features:
        lang_field = 'lang_id'
    
    if lang_field:
        print(f"Found language field: '{lang_field}'")
        
        # Count examples per language
        print("Analyzing language distribution...")
        languages = [item[lang_field] for item in dataset]
        lang_counts = Counter(languages)
        
        print(f"\nTotal examples in dataset: {len(dataset):,}")
        print(f"Number of languages: {len(lang_counts)}")
        print("\nOriginal examples per language:")
        for lang, count in sorted(lang_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(dataset)) * 100
            print(f"  {lang}: {count:,} examples ({percentage:.2f}%)")
    elif 'sectionText' in dataset.features:
        # This is a wstp.* dataset - it's single language (language is in the dataset name)
        print("⚠️  This appears to be a single-language dataset (wstp.* format)")
        print("   The dataset name indicates the language (e.g., wstp.hi = Hindi)")
        print("   For multilingual training, you need to:")
        print("   1. Load from Google Drive with combined dataset from multiple languages")
        print("   2. Or download multiple language subsets and combine them")
        print(f"\n   Current dataset: {len(dataset):,} examples (single language)")
        print("   All examples will be used for training.")
        
        # Sample 30k examples per language
        print(f"\n{'='*60}")
        print(f"Sampling {MAX_SAMPLES_PER_LANGUAGE:,} examples per language...")
        print(f"{'='*60}")
        
        # Group examples by language
        lang_groups = defaultdict(list)
        for i, item in enumerate(dataset):
            lang = item[lang_field]
            lang_groups[lang].append(i)
        
        # Sample from each language
        sampled_indices = []
        final_lang_counts = {}
        
        for lang, indices in lang_groups.items():
            if len(indices) > MAX_SAMPLES_PER_LANGUAGE:
                # Randomly sample MAX_SAMPLES_PER_LANGUAGE examples
                sampled = random.sample(indices, MAX_SAMPLES_PER_LANGUAGE)
                sampled_indices.extend(sampled)
                final_lang_counts[lang] = MAX_SAMPLES_PER_LANGUAGE
                print(f"  {lang}: Sampled {MAX_SAMPLES_PER_LANGUAGE:,} from {len(indices):,} available")
            else:
                # Use all examples if less than MAX_SAMPLES_PER_LANGUAGE
                sampled_indices.extend(indices)
                final_lang_counts[lang] = len(indices)
                print(f"  {lang}: Using all {len(indices):,} examples (less than {MAX_SAMPLES_PER_LANGUAGE:,})")
        
        # Create new dataset with sampled examples
        dataset = dataset.select(sampled_indices)
        
        print(f"\n{'='*60}")
        print("Final Dataset After Sampling:")
        print(f"{'='*60}")
        print(f"Total examples for training: {len(dataset):,}")
        print(f"Number of languages: {len(final_lang_counts)}")
        print("\nFinal examples per language:")
        for lang, count in sorted(final_lang_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(dataset)) * 100
            print(f"  {lang}: {count:,} examples ({percentage:.2f}%)")
        
    else:
        print("⚠️  No explicit language field found in dataset.")
        print("Available fields:", list(dataset.features.keys()))
        print("\nCannot sample per language - using all examples.")
        print("All examples in the dataset will be used for training.")
        
except Exception as e:
    print(f"❌ Error analyzing language distribution: {e}")
    print("Using all examples in the dataset for training.")
    import traceback
    traceback.print_exc()

print("="*60)

# ============================================================================
# Multilingual Prompt Template and Formatting Function
# ============================================================================
# Multilingual prompt template for Indian languages
multilingual_prompt = """You are a helpful multilingual assistant capable of understanding and responding in multiple Indian languages including Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Odia, Assamese, Urdu, and more.

Please respond to the following in the same language as the input:

{}
"""
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
  """Format the dataset for training.
  
  This function adapts to different dataset structures.
  Common field names for multilingual datasets:
  - 'instruction' or 'input' or 'question' or 'prompt' for the input
  - 'output' or 'response' or 'answer' or 'target' for the response
  - 'text' for pre-formatted text
  - 'language' or 'lang' for language identifier (optional)
  - 'sectionText' and 'correctTitle' for wiki section title prediction tasks
  """
  texts = []
  
  # Try to detect the dataset structure
  if 'text' in examples:
    # If dataset already has formatted text
    for text in examples['text']:
      texts.append(text + EOS_TOKEN)
  elif 'sectionText' in examples and 'correctTitle' in examples:
    # Wiki Section Title Prediction format (wstp.* datasets)
    # Format: sectionText -> correctTitle (which is a key like 'titleA', 'titleB', etc.)
    section_texts = examples['sectionText']
    correct_title_keys = examples['correctTitle']
    title_as = examples.get('titleA', [''] * len(section_texts))
    title_bs = examples.get('titleB', [''] * len(section_texts))
    title_cs = examples.get('titleC', [''] * len(section_texts))
    title_ds = examples.get('titleD', [''] * len(section_texts))
    
    for i, (section_text, correct_key) in enumerate(zip(section_texts, correct_title_keys)):
      # Get the actual title based on the correct key
      if correct_key == 'titleA':
        correct_title = title_as[i] if i < len(title_as) else ''
      elif correct_key == 'titleB':
        correct_title = title_bs[i] if i < len(title_bs) else ''
      elif correct_key == 'titleC':
        correct_title = title_cs[i] if i < len(title_cs) else ''
      elif correct_key == 'titleD':
        correct_title = title_ds[i] if i < len(title_ds) else ''
      else:
        correct_title = ''
      
      # Format as instruction-output pair
      instruction = f"Given the following text section, provide an appropriate title:\n\n{section_text}"
      output = correct_title
      text = multilingual_prompt.format(f"{instruction}\n\n{output}") + EOS_TOKEN
      texts.append(text)
  elif 'instruction' in examples and 'output' in examples:
    # Common format: instruction-output pairs
    for instruction, output in zip(examples['instruction'], examples['output']):
      text = multilingual_prompt.format(f"{instruction}\n\n{output}") + EOS_TOKEN
      texts.append(text)
  elif 'input' in examples and 'output' in examples:
    # Alternative format: input-output pairs
    for inp, out in zip(examples['input'], examples['output']):
      text = multilingual_prompt.format(f"{inp}\n\n{out}") + EOS_TOKEN
      texts.append(text)
  elif 'question' in examples and 'answer' in examples:
    # Q&A format
    for question, answer in zip(examples['question'], examples['answer']):
      text = multilingual_prompt.format(f"Question: {question}\n\nAnswer: {answer}") + EOS_TOKEN
      texts.append(text)
  else:
    # Fallback: try to use first two text-like columns
    keys = list(examples.keys())
    text_keys = [k for k in keys if 'text' in k.lower() or 'content' in k.lower() or 'input' in k.lower()]
    if len(text_keys) >= 2:
      col1, col2 = text_keys[0], text_keys[1]
      for val1, val2 in zip(examples[col1], examples[col2]):
        text = multilingual_prompt.format(f"{val1}\n\n{val2}") + EOS_TOKEN
        texts.append(text)
    elif len(text_keys) == 1:
      # Single text field - use as is
      for text in examples[text_keys[0]]:
        texts.append(text + EOS_TOKEN)
    else:
      raise ValueError(f"Could not detect dataset format. Available columns: {keys}")

  return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

# ============================================================================
# Model Configuration
# ============================================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for longer context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# ============================================================================
# Training Setup
# ============================================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,  # Number of processors to use for processing the dataset
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,  # The batch size per GPU/TPU core
        gradient_accumulation_steps=4,  # Number of steps to perform before each gradient accumulation
        warmup_steps=5,  # Few updates with low learning rate before actual training
        max_steps=60,  # Specifies the total number of training steps (batches) to run.
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",  # Optimizer
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Use this for WandB etc for observability
    ),
)

# ============================================================================
# Training
# ============================================================================
trainer.train()

# ============================================================================
# Save Model
# ============================================================================
model.save_pretrained("lora_model")  # Local saving
tokenizer.save_pretrained("lora_model")

# Optional: Merge and save for inference
# model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit",)
# model.save_pretrained_merged("model", tokenizer, save_method="merged_4bit",)
# model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_4bit", token="")
# model.push_to_hub_merged("hf/model", tokenizer, save_method="merged_16bit", token="")

