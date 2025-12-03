#!/usr/bin/env python3
"""
🧪 Test Script for Fine-tuned Multilingual Model

This script loads the fine-tuned model and tests generation capabilities
across all Indian languages the model was trained on:
- Hindi (hi), Tamil (ta), Telugu (te), Malayalam (ml), Kannada (kn)
- Gujarati (gu), Marathi (mr), Bengali (bn), Assamese (as), Odia (or), Punjabi (pa)

USAGE:
    python test_finetuned_model.py [--model_path MODEL_PATH] [--base_model BASE_MODEL]

EXAMPLES:
    # Use default paths
    python test_finetuned_model.py
    
    # Specify custom model path
    python test_finetuned_model.py --model_path ./outputs/lora_model
    
    # Specify both model path and base model
    python test_finetuned_model.py --model_path ./lora_model --base_model unsloth/Llama-3.2-3B-Instruct-bnb-4bit
"""

import torch
import argparse
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported

# ============================================================================
# Parse Command Line Arguments
# ============================================================================
parser = argparse.ArgumentParser(description="Test fine-tuned multilingual model")
parser.add_argument(
    "--model_path",
    type=str,
    default="lora_model",
    help="Path to the saved fine-tuned model (default: 'lora_model')"
)
parser.add_argument(
    "--base_model",
    type=str,
    default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    help="Base model name used for training (default: 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit')"
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=256,
    help="Maximum number of tokens to generate (default: 256)"
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="Temperature for generation (default: 0.7)"
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.9,
    help="Top-p (nucleus) sampling parameter (default: 0.9)"
)
parser.add_argument(
    "--repetition_penalty",
    type=float,
    default=1.2,
    help="Repetition penalty to prevent repetitive outputs (default: 1.2)"
)
parser.add_argument(
    "--min_length",
    type=int,
    default=10,
    help="Minimum length of generated text (default: 10)"
)

# Use parse_known_args() to ignore unknown arguments (useful in Colab/Jupyter)
# This prevents errors when Colab passes kernel-related arguments
args, unknown = parser.parse_known_args()

# Warn about unknown arguments if any (but don't fail)
if unknown:
    print(f"Note: Ignoring unknown arguments: {unknown}")

# ============================================================================
# Model Loading Configuration
# ============================================================================
MODEL_PATH = args.model_path  # Path to the saved fine-tuned model
BASE_MODEL = args.base_model  # Base model used for training
max_seq_length = 2048
dtype = None  # None for auto detection
load_in_4bit = True  # Use 4bit quantization

# ============================================================================
# Load Fine-tuned Model
# ============================================================================
print("="*60)
print("Loading Fine-tuned Model")
print("="*60)

import os

# Check if model path exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model path '{MODEL_PATH}' not found. "
        f"Please ensure you have run the training script and the model is saved."
    )

model_loaded = False

# Method 1: Try loading as a LoRA adapter (most common case)
try:
    print(f"Method 1: Loading base model and LoRA adapter...")
    print(f"  Base model: {BASE_MODEL}")
    
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    
    # Load LoRA adapter using PEFT
    from peft import PeftModel
    print(f"  Loading LoRA adapter from: {MODEL_PATH}")
    model = PeftModel.from_pretrained(model, MODEL_PATH)
    
    # Enable inference mode
    FastLanguageModel.for_inference(model)
    
    print("✓ Model loaded successfully (LoRA adapter)!")
    model_loaded = True
    print("="*60)
    
except Exception as e:
    print(f"  ❌ Method 1 failed: {str(e)[:200]}")
    print("\nTrying Method 2...")

# Method 2: Try loading directly (if it's a merged model or full model)
if not model_loaded:
    try:
        print(f"Method 2: Loading model directly from: {MODEL_PATH}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_PATH,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        print("✓ Model loaded successfully (direct load)!")
        model_loaded = True
        print("="*60)
    except Exception as e:
        print(f"  ❌ Method 2 failed: {str(e)[:200]}")

# Method 3: Try loading base model and then applying adapter manually
if not model_loaded:
    try:
        print(f"Method 3: Loading base model and applying adapter manually...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )
        
        # Try to load adapter weights manually
        from peft import PeftConfig
        config = PeftConfig.from_pretrained(MODEL_PATH)
        from peft import get_peft_model, LoraConfig
        
        # Recreate PEFT config
        peft_config = LoraConfig(
            r=config.r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.bias,
            task_type=config.task_type,
        )
        
        model = get_peft_model(model, peft_config)
        model.load_adapter(MODEL_PATH)
        FastLanguageModel.for_inference(model)
        
        print("✓ Model loaded successfully (manual adapter load)!")
        model_loaded = True
        print("="*60)
    except Exception as e:
        print(f"  ❌ Method 3 failed: {str(e)[:200]}")

if not model_loaded:
    raise RuntimeError(
        f"Failed to load model from '{MODEL_PATH}'. "
        f"Please check:\n"
        f"1. The model was saved correctly after training\n"
        f"2. The MODEL_PATH is correct\n"
        f"3. You have the required dependencies installed"
    )

# ============================================================================
# Multilingual Prompt Template (Same as Training)
# ============================================================================
multilingual_prompt = """You are a helpful multilingual assistant capable of understanding and responding in multiple Indian languages including Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Odia, Assamese, Urdu, and more.

Please respond to the following in the same language as the input:

{}
"""

# ============================================================================
# Test Cases for Each Language
# ============================================================================
test_cases = {
    "Hindi (hi)": [
        {
            "task": "Title Generation",
            "input": "भारत एक विविधतापूर्ण देश है जहाँ कई भाषाएँ बोली जाती हैं। यहाँ विभिन्न संस्कृतियाँ और परंपराएँ हैं।",
            "expected_type": "title"
        },
        {
            "task": "Question Answering",
            "input": "भारत की राजधानी क्या है?",
            "expected_type": "qa"
        },
        {
            "task": "General Conversation",
            "input": "मशीन लर्निंग क्या है?",
            "expected_type": "general"
        }
    ],
    "Tamil (ta)": [
        {
            "task": "Title Generation",
            "input": "தமிழ்நாடு தென்னிந்தியாவில் உள்ள ஒரு மாநிலம். இது பழமையான மொழி மற்றும் கலாச்சாரத்தைக் கொண்டுள்ளது.",
            "expected_type": "title"
        },
        {
            "task": "Question Answering",
            "input": "சென்னையின் முக்கிய கடற்கரை எது?",
            "expected_type": "qa"
        },
        {
            "task": "General Conversation",
            "input": "கணினி கற்றல் என்றால் என்ன?",
            "expected_type": "general"
        }
    ],
    "Telugu (te)": [
        {
            "task": "Title Generation",
            "input": "తెలుగు భారతదేశంలోని ప్రధాన భాషలలో ఒకటి. ఇది ఆంధ్రప్రదేశ్ మరియు తెలంగాణ రాష్ట్రాలలో మాట్లాడబడుతుంది.",
            "expected_type": "title"
        },
        {
            "task": "Question Answering",
            "input": "హైదరాబాద్ ఏ రాష్ట్రంలో ఉంది?",
            "expected_type": "qa"
        },
        {
            "task": "General Conversation",
            "input": "కృత్రిమ మేధస్సు అంటే ఏమిటి?",
            "expected_type": "general"
        }
    ],
    "Malayalam (ml)": [
        {
            "task": "Title Generation",
            "input": "കേരളം ഇന്ത്യയുടെ തെക്കുപടിഞ്ഞാറൻ സംസ്ഥാനമാണ്. ഇത് സമൃദ്ധമായ സംസ്കാരവും സാഹിത്യവും ഉള്ളതാണ്.",
            "expected_type": "title"
        },
        {
            "task": "Question Answering",
            "input": "കേരളത്തിന്റെ തലസ്ഥാനം എന്താണ്?",
            "expected_type": "qa"
        },
        {
            "task": "General Conversation",
            "input": "ഡാറ്റാ സയൻസ് എന്താണ്?",
            "expected_type": "general"
        }
    ],
    "Kannada (kn)": [
        {
            "task": "Title Generation",
            "input": "ಕರ್ನಾಟಕ ದಕ್ಷಿಣ ಭಾರತದಲ್ಲಿರುವ ಒಂದು ರಾಜ್ಯ. ಇದು ಸಂಪನ್ಮೂಲಗಳಿಂದ ಸಮೃದ್ಧವಾಗಿದೆ ಮತ್ತು ತಂತ್ರಜ್ಞಾನ ಕೇಂದ್ರವಾಗಿದೆ.",
            "expected_type": "title"
        },
        {
            "task": "Question Answering",
            "input": "ಬೆಂಗಳೂರಿನ ಮುಖ್ಯ ಉದ್ಯಾನವನ ಯಾವುದು?",
            "expected_type": "qa"
        },
        {
            "task": "General Conversation",
            "input": "ನ್ಯೂರಲ್ ನೆಟ್ವರ್ಕ್ ಎಂದರೇನು?",
            "expected_type": "general"
        }
    ],
    "Gujarati (gu)": [
        {
            "task": "Title Generation",
            "input": "ગુજરાત પશ્ચિમ ભારતમાં આવેલું એક રાજ્ય છે. તે ઉદ્યોગ અને વેપાર માટે પ્રખ્યાત છે.",
            "expected_type": "title"
        },
        {
            "task": "Question Answering",
            "input": "અમદાવાદ કયા રાજ્યમાં આવેલું છે?",
            "expected_type": "qa"
        },
        {
            "task": "General Conversation",
            "input": "ડીપ લર્નિંગ શું છે?",
            "expected_type": "general"
        }
    ],
    "Marathi (mr)": [
        {
            "task": "Title Generation",
            "input": "महाराष्ट्र पश्चिम भारतातील एक राज्य आहे. येथे समृद्ध इतिहास आणि संस्कृती आहे.",
            "expected_type": "title"
        },
        {
            "task": "Question Answering",
            "input": "मुंबईची लोकसंख्या किती आहे?",
            "expected_type": "qa"
        },
        {
            "task": "General Conversation",
            "input": "नेचरल लँग्वेज प्रोसेसिंग म्हणजे काय?",
            "expected_type": "general"
        }
    ],
    "Bengali (bn)": [
        {
            "task": "Title Generation",
            "input": "পশ্চিমবঙ্গ পূর্ব ভারতের একটি রাজ্য। এটি সাহিত্য ও সংস্কৃতির জন্য বিখ্যাত।",
            "expected_type": "title"
        },
        {
            "task": "Question Answering",
            "input": "কলকাতা কোন রাজ্যের রাজধানী?",
            "expected_type": "qa"
        },
        {
            "task": "General Conversation",
            "input": "মেশিন লার্নিং কি?",
            "expected_type": "general"
        }
    ],
    "Assamese (as)": [
        {
            "task": "Title Generation",
            "input": "আসাম উত্তৰ-পূব ভাৰতৰ এখন ৰাজ্য। ইয়াৰ প্ৰাকৃতিক সম্পদ আৰু সাংস্কৃতিক ঐতিহ্য আছে।",
            "expected_type": "title"
        },
        {
            "task": "Question Answering",
            "input": "গুৱাহাটী কোন ৰাজ্যৰ ৰাজধানী?",
            "expected_type": "qa"
        },
        {
            "task": "General Conversation",
            "input": "কৃত্ৰিম বুদ্ধিমত্তা কি?",
            "expected_type": "general"
        }
    ],
    "Odia (or)": [
        {
            "task": "Title Generation",
            "input": "ଓଡ଼ିଶା ପୂର୍ବ ଭାରତରେ ଅବସ୍ଥିତ ଏକ ରାଜ୍ୟ। ଏହାର ସମୃଦ୍ଧ ସଂସ୍କୃତି ଏବଂ ଇତିହାସ ଅଛି।",
            "expected_type": "title"
        },
        {
            "task": "Question Answering",
            "input": "ଭୁବନେଶ୍ୱର କେଉଁ ରାଜ୍ୟର ରାଜଧାନୀ?",
            "expected_type": "qa"
        },
        {
            "task": "General Conversation",
            "input": "କମ୍ପ୍ୟୁଟର ବିଜ୍ଞାନ କଣ?",
            "expected_type": "general"
        }
    ],
    "Punjabi (pa)": [
        {
            "task": "Title Generation",
            "input": "ਪੰਜਾਬ ਉੱਤਰੀ ਭਾਰਤ ਵਿੱਚ ਸਥਿਤ ਇੱਕ ਰਾਜ ਹੈ। ਇਸ ਦੀ ਖੇਤੀਬਾੜੀ ਅਤੇ ਸੱਭਿਆਚਾਰਕ ਵਿਰਾਸਤ ਮਸ਼ਹੂਰ ਹੈ।",
            "expected_type": "title"
        },
        {
            "task": "Question Answering",
            "input": "ਅੰਮ੍ਰਿਤਸਰ ਕਿਹੜੇ ਰਾਜ ਵਿੱਚ ਹੈ?",
            "expected_type": "qa"
        },
        {
            "task": "General Conversation",
            "input": "ਡਾਟਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕੀ ਹੈ?",
            "expected_type": "general"
        }
    ]
}

# ============================================================================
# Generation Function
# ============================================================================
def generate_response(prompt, max_new_tokens=None, temperature=None, top_p=None, task_type="general"):
    """Generate response from the model with improved parameters."""
    # Use command-line arguments if not provided
    if max_new_tokens is None:
        max_new_tokens = args.max_new_tokens
    if temperature is None:
        temperature = args.temperature
    if top_p is None:
        top_p = args.top_p
    
    # Adjust parameters based on task type
    if task_type == "title":
        # For titles, use lower temperature and shorter max tokens
        temperature = min(temperature, 0.5)
        max_new_tokens = min(max_new_tokens, 50)
    elif task_type == "qa":
        # For Q&A, use moderate temperature
        temperature = min(temperature, 0.6)
    
    # Format the prompt using the same template as training
    formatted_prompt = multilingual_prompt.format(prompt)
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length
    ).to(model.device)
    
    # Generate with improved parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_length=args.min_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=args.repetition_penalty,  # Prevent repetition
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part (remove the input prompt)
    if formatted_prompt in generated_text:
        response = generated_text.split(formatted_prompt)[-1].strip()
    else:
        response = generated_text.strip()
    
    # Clean up response - remove any remaining prompt artifacts
    if response.startswith("Please respond to the following"):
        # Try to extract the actual response
        parts = response.split("\n\n")
        if len(parts) > 1:
            response = "\n\n".join(parts[1:]).strip()
    
    return response

# ============================================================================
# Run Tests
# ============================================================================
print("\n" + "="*60)
print("Running Generation Tests")
print("="*60)

results = {}

for language, test_list in test_cases.items():
    print(f"\n{'='*60}")
    print(f"Testing: {language}")
    print(f"{'='*60}")
    
    language_results = []
    
    for i, test_case in enumerate(test_list, 1):
        print(f"\n[Test {i}/{len(test_list)}] {test_case['task']}")
        print(f"Input: {test_case['input'][:100]}..." if len(test_case['input']) > 100 else f"Input: {test_case['input']}")
        print("-" * 60)
        
        try:
            # For title generation tasks, format similar to training with explicit instruction
            if test_case['expected_type'] == 'title':
                prompt = f"Given the following text section, provide an appropriate title (1-5 words only):\n\n{test_case['input']}"
            elif test_case['expected_type'] == 'qa':
                # For Q&A, make it explicit that we want a direct answer
                prompt = f"Answer the following question directly and concisely:\n\n{test_case['input']}"
            else:
                prompt = test_case['input']
            
            response = generate_response(
                prompt, 
                max_new_tokens=256,
                task_type=test_case['expected_type']
            )
            
            print(f"Generated Response:")
            print(f"{response}")
            print("-" * 60)
            
            language_results.append({
                "task": test_case['task'],
                "input": test_case['input'],
                "output": response,
                "status": "success"
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            language_results.append({
                "task": test_case['task'],
                "input": test_case['input'],
                "output": None,
                "status": f"error: {str(e)}"
            })
    
    results[language] = language_results

# ============================================================================
# Summary Report
# ============================================================================
print("\n" + "="*60)
print("Test Summary Report")
print("="*60)

total_tests = 0
successful_tests = 0
failed_tests = 0

for language, language_results in results.items():
    lang_success = sum(1 for r in language_results if r['status'] == 'success')
    lang_total = len(language_results)
    total_tests += lang_total
    successful_tests += lang_success
    failed_tests += (lang_total - lang_success)
    
    status_icon = "✓" if lang_success == lang_total else "⚠"
    print(f"\n{status_icon} {language}: {lang_success}/{lang_total} tests passed")

print(f"\n{'='*60}")
print(f"Overall: {successful_tests}/{total_tests} tests passed ({100*successful_tests/total_tests:.1f}%)")
print(f"{'='*60}")

# ============================================================================
# Save Results
# ============================================================================
import json
from datetime import datetime

output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"\n✓ Test results saved to: {output_file}")

print("\n" + "="*60)
print("Testing Complete!")
print("="*60)

