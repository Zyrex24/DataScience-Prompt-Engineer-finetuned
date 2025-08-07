# DataScience Prompt Engineer

A fine-tuned, instruction-following language model for generating Python code and assisting with data science prompts, workflows, and analysis. This project includes the full process from dataset preparation and model training to deployment using Hugging Face Transformers, PEFT/Unsloth, and robust quantized (4-bit) inference support.

---

## Features

- Custom dataset preparation and tokenization, focused on data science-related tasks and responses.
- Complete, scriptable training pipeline (using Unsloth/PEFT and Hugging Face Trainer).
- Stepwise logging of training loss for model evaluation and reproducibility.
- Support for merging LoRA/adapter weights into a full, quantized model for easy deployment (Python, Colab, Spaces).
- Examples showing how to fine-tune, merge, and serve your own instruction-tuned model for code generation.

## Contents

- Preprocessing scripts for dataset formatting and tokenization.
- Training loop and configuration for efficient, low-RAM fine-tuning.
- Workflow and scripts for saving and pushing both adapters and merged, ready-to-infer models to Hugging Face Model Hub.
- Example inference and Gradio UI web app for demos or production.

## Installation & Usage

1. **Clone the repository:**
git clone https://github.com/Zyrex24/DataScience-Prompt-Engineer-finetuned.git
cd DataScience-Prompt-Engineer-finetuned

2. **Install dependencies:**
pip install -r requirements.txt

3. **Training / Fine-tuning:**
- Use the included notebook or `datasciencepromptengineer.py` script to fine-tune the base model using your filtered dataset.
- Adapter weights and tokenizer are saved and pushed to the Hugging Face Hub.

4. **Merge and Prepare Model for Deployment:**
- After training and pushing your adapter, use the provided merge-and-push script to produce a standalone model repo (see notebook for step-by-step guide).
- The merged repo (such as `ZyrexAN/DataScience-PromptEngineer-merged`) includes `pytorch_model.bin` or `model.safetensors`, tokenizer, and configs for instant inference compatibility.

5. **Run Locally with Gradio:**

- For longer outputs, increase `max_new_tokens`.
- For advanced UI, see examples in the repo.

## Model Hub

- Find the baseline adapter and merged, ready-to-infer model at:
- [`ZyrexAN/DataScience-PromptEngineer`](https://huggingface.co/ZyrexAN/DataScience-PromptEngineer)
- [`ZyrexAN/DataScience-PromptEngineer-merged`](https://huggingface.co/ZyrexAN/DataScience-PromptEngineer-merged)

## Advanced Topics

- **GPU Required** for 7B models
- **Troubleshooting:** If you see output cutoff, increase `max_new_tokens`. For missing package errors, check `requirements.txt` and ensure you restart Python after new installs.
- **Merging Adapters:** For a standalone pipeline-compatible model, see notebook step “Merge and Push Merged Model”.

## Contributing

Pull requests and issue reports are welcome. Please create an issue if you’d like to suggest features, request fixes, or share ideas!

---
