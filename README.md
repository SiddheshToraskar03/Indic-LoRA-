# IndicLoRA: Multilingual Indian Languages Fine-tuning with Unsloth

<p align="center">
  <img src="https://img.shields.io/badge/Language-Multilingual-blue" alt="Multilingual">
  <img src="https://img.shields.io/badge/Framework-Unsloth-orange" alt="Unsloth Framework">
  <img src="https://img.shields.io/badge/Model-Llama3.2--3B-green" alt="Llama3.2-3B">
  <img src="https://img.shields.io/badge/License-Apache--2.0-yellow" alt="Apache-2.0 License">
</p>

## Overview

IndicLoRA is a specialized project designed to fine-tune large language models for Indian languages using the efficient Unsloth framework. This project leverages Low-Rank Adaptation (LoRA) techniques to adapt the Llama-3.2-3B-Instruct model for understanding and generating text in multiple Indian languages while maintaining efficiency and performance.

## Supported Languages

The model is trained to work with 11+ major Indian languages:

- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Kannada (kn)
- Malayalam (ml)
- Bengali (bn)
- Gujarati (gu)
- Marathi (mr)
- Punjabi (pa)
- Odia (or)
- Assamese (as)

## Key Features

- **Efficient Fine-tuning**: Uses Unsloth framework for faster and more memory-efficient training
- **LoRA Adaptation**: Implements Low-Rank Adaptation for lightweight model customization
- **Multilingual Support**: Comprehensive coverage of major Indian languages
- **Task Versatility**: Capable of handling various NLP tasks including question answering and title generation
- **Quantized Model**: Uses 4-bit quantization to reduce memory requirements

## Project Structure

```
├── 2_unsloth_finetuning.py        # Main fine-tuning script
├── download_ai4bharat_dataset.py  # Dataset downloader
├── test_finetuned_model.py        # Model testing and evaluation
├── output_quality_assessment.md   # Detailed quality assessment report
├── lora_model/                    # Trained LoRA adapter files
├── ai4bharat_dataset/             # Downloaded dataset (when available)
└── outputs/                       # Training outputs and logs
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SiddheshToraskar03/Indic-LoRA-.git
   cd Indic-LoRA-
   ```

2. Install required dependencies:
   ```bash
   pip install -q unsloth
   pip install -q --force-reinstall --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
   pip install torch datasets trl transformers peft huggingface_hub
   ```

## Usage

### 1. Dataset Preparation

First, download the AI4Bharat dataset:
```bash
python download_ai4bharat_dataset.py
```

Upload the generated `ai4bharat_dataset` folder to Google Drive if using Colab.

### 2. Fine-tuning

Run the fine-tuning script:
```bash
python 2_unsloth_finetuning.py
```

For Google Colab, update the script to set `LOAD_FROM_DRIVE = True` and configure the correct Drive path.

### 3. Testing

Evaluate the fine-tuned model:
```bash
python test_finetuned_model.py
```

You can customize generation parameters:
```bash
python test_finetuned_model.py --temperature 0.5 --max_new_tokens 128
```

## Model Performance

The model demonstrates varying performance across different languages and tasks:

| Language | Title Generation | Question Answering | Overall |
|----------|------------------|-------------------|---------|
| Telugu   | ✅ Good          | ✅ Good           | ✅ Good |
| Bengali  | ⚠️ Moderate      | ✅ Good           | ⚠️ Moderate |
| Hindi    | ⚠️ Moderate      | ⚠️ Moderate       | ⚠️ Moderate |
| Others   | ⚠️ Moderate      | ⚠️ Moderate       | ⚠️ Moderate |

See [output_quality_assessment.md](output_quality_assessment.md) for detailed performance analysis.

## Known Issues & Limitations

1. **Task Understanding**: Model sometimes confuses task types (e.g., generates explanations instead of titles)
2. **Repetition**: Some languages show repetition issues in generated text
3. **Empty Responses**: Occasional empty responses for certain language-task combinations
4. **Factual Accuracy**: Some factual inaccuracies in responses

## Recommendations for Improvement

1. **Increase Training Steps**: Current implementation uses only 60 steps; increasing to 200-500 could improve performance
2. **Enhanced Prompt Engineering**: More explicit task instructions in prompts
3. **Better Dataset**: Include more diverse training examples with balanced language representation
4. **Improved Generation Parameters**: Use lower temperature and repetition penalties

## Contributing

Contributions to improve the model's performance across languages are welcome. Areas for contribution include:

- Adding support for more Indian languages
- Improving the training dataset quality
- Enhancing prompt templates for better task understanding
- Optimizing generation parameters for specific languages

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for the efficient fine-tuning framework
- [AI4Bharat](https://ai4bharat.org/) for the multilingual datasets
- Meta for the Llama-3.2-3B-Instruct base model

## Contact

For questions or feedback, please open an issue on this repository.