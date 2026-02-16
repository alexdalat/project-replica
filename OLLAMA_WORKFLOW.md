# Ollama Export Workflow

This directory contains scripts to automate the workflow from fine-tuned model to Ollama model creation.

## Workflow

### 1. Export your data source

Choose the appropriate export script for your data source:

**Gmail:**
```bash
./export_gmail.sh data_$USER 5
```

**Google Docs:**
```bash
./export_gdocs.sh data_$USER 5
```

**iMessages:**
```bash
./export_imessages.sh data_$USER
```

### 2. Fine-tune the model

```bash
python3 finetune.py data_$USER/final_gmail.jsonl models/$USER \
    --dataset_limit=500 \
    --epochs=1 \
    --batch_size=1
```

This will create a model directory like: `models/$USER/gmail_2025-11-11_12-33`

### 3. Export to Ollama

```bash
./ollama_export.sh models/$USER/gmail_2025-11-11_12-33
```

This single script will:
- Export the LoRA adapter to GGUF format
- Create an Ollama Modelfile with proper configuration

### 4. Create the Ollama model

```bash
cd ~/code/other/ollama-webui/models
ollama create alex_gmail -f Modelfile.alex_gmail
```

## ollama_export.sh

The `ollama_export.sh` script is a universal export tool that works with any fine-tuned model directory. It:
- Takes a fine-tuned model directory as input
- Exports the LoRA adapter to GGUF format using `export_lora.py`
- Creates a Modelfile using `create_modelfile.py`

**Usage:**
```bash
./ollama_export.sh <model_directory> [ollama_models_dir]

# Examples:
./ollama_export.sh models/alex/gmail_2025-11-11_12-33
./ollama_export.sh models/alex/gdocs_2025-11-11_14-22 ~/custom/ollama/models
```

## create_modelfile.py

This script automatically:
- Reads the adapter configuration to determine the base model
- Maps HuggingFace model names to Ollama model names
- Extracts training parameters (like max_length) to set num_ctx
- Creates a properly formatted Modelfile with:
  - Base model reference
  - Adapter path (absolute path to the GGUF file)
  - Context window size (num_ctx)
  - Repeat penalty parameter

### Supported Base Models

The script automatically maps these HuggingFace models to Ollama equivalents:
- `mistralai/Mistral-7B-Instruct-v0.3` → `mistral:instruct`
- `mistralai/Mistral-7B-Instruct-v0.2` → `mistral:instruct`
- `mistralai/Mistral-7B-Instruct-v0.1` → `mistral:instruct`
- `meta-llama/Llama-2-7b-chat-hf` → `llama2:7b-chat`
- `meta-llama/Llama-2-13b-chat-hf` → `llama2:13b-chat`

Default fallback: `mistral:instruct`

## Example Modelfile Output

```
FROM mistral:instruct
ADAPTER /home/alex/code/py/imsg_llm/models/alex/gmail_2025-11-11_12-33/out/lora_adaptor-f16.gguf

PARAMETER num_ctx 2048
PARAMETER repeat_penalty 1.15
```

## Troubleshooting

### "No LoRA adapter GGUF found in out/ directory"
Run `export_lora.py` first:
```bash
python3 export_lora.py models/$USER/gmail_2025-11-11_12-33
```

### "adapter_config.json not found"
Make sure you're pointing to a valid fine-tuned model directory that contains the LoRA adapter files.

### Model doesn't work in Ollama
Ensure the base model referenced in the Modelfile is installed in Ollama:
```bash
ollama pull mistral:instruct
```
