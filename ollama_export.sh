#!/bin/bash
# Export a fine-tuned LoRA model to Ollama format
# Usage: ./ollama_export.sh models/alex/gmail_2025-11-11_12-33 [ollama_models_dir]

set -e  # Exit on error

# Check arguments
if [ -z "$1" ]; then
	echo "Usage: $0 <model_directory> [ollama_models_dir]"
	echo "Example: $0 models/alex/gmail_2025-11-11_12-33 ~/code/other/ollama-webui/models"
	echo ""
	echo "This script will:"
	echo "  1. Export the LoRA adapter to GGUF format"
	echo "  2. Create an Ollama Modelfile"
	exit 1
fi

MODEL_DIR="$1"
OLLAMA_MODELS_DIR=${2:-"$HOME/code/other/ollama-webui/models"}

# Check that model directory exists
if [ ! -d "$MODEL_DIR" ]; then
	echo "Error: Model directory does not exist: $MODEL_DIR"
	exit 1
fi

# Check that it looks like a valid model directory
if [ ! -f "$MODEL_DIR/adapter_config.json" ]; then
	echo "Error: $MODEL_DIR does not appear to be a valid LoRA model directory"
	echo "       (adapter_config.json not found)"
	exit 1
fi

echo "========================================="
echo "Exporting model to Ollama format"
echo "========================================="
echo "Model directory: $MODEL_DIR"
echo "Ollama models directory: $OLLAMA_MODELS_DIR"
echo ""

echo "========================================="
echo "Step 1: Exporting LoRA adapter to GGUF"
echo "========================================="
python3 export_lora.py "$MODEL_DIR"

if [ $? -ne 0 ]; then
	echo "Error during LoRA export. Exiting."
	exit 1
fi

echo "✓ LoRA export complete!"
echo ""

echo "========================================="
echo "Step 2: Creating Ollama Modelfile"
echo "========================================="
python3 create_modelfile.py "$MODEL_DIR" "$OLLAMA_MODELS_DIR"

if [ $? -ne 0 ]; then
	echo "Error during Modelfile creation. Exiting."
	exit 1
fi

echo "✓ Modelfile created!"
echo ""

# Extract model info for final instructions
USERNAME=$(basename "$(dirname "$MODEL_DIR")")
DATASET_TYPE=$(basename "$MODEL_DIR" | cut -d'_' -f1)

echo "========================================="
echo "✓ Export complete!"
echo "========================================="
echo ""
echo "Model directory: $MODEL_DIR"
echo "Modelfile location: $OLLAMA_MODELS_DIR/Modelfile.${USERNAME}_${DATASET_TYPE}"
echo ""
echo "To create the model in Ollama, run:"
echo "  cd $OLLAMA_MODELS_DIR"
echo "  ollama create ${USERNAME}_${DATASET_TYPE} -f Modelfile.${USERNAME}_${DATASET_TYPE}"
echo ""
