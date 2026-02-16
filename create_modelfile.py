#!/usr/bin/env python3
"""
Create an Ollama Modelfile from a fine-tuned LoRA adapter.

Usage:
    python3 create_modelfile.py models/alex/gmail_2025-11-11_12-33 ~/code/other/ollama-webui/models
"""

import argparse
import json
from pathlib import Path
from typing import Optional


def find_adapter_config(model_dir: Path) -> Optional[Path]:
    """Find the adapter_config.json in the model directory or its subdirectories."""
    candidates = list(model_dir.glob("adapter_config.json")) + list(
        model_dir.glob("**/adapter_config.json")
    )
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: len(p.parts))[0]


def read_base_model_info(config_path: Path) -> dict:
    """Extract base model information from adapter_config.json."""
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Extract base model name
    base_model_keys = ["base_model_name_or_path", "base_model_name", "base_model"]
    base_model = None
    for key in base_model_keys:
        if key in config and config[key]:
            base_model = config[key]
            break
    
    return {
        "base_model": base_model,
        "config": config,
    }


def determine_base_model_name(base_model_path: str) -> str:
    """Convert HuggingFace model path to Ollama model name."""
    # Map common HF models to Ollama equivalents
    model_mapping = {
        "mistralai/Mistral-7B-Instruct-v0.3": "mistral:instruct",
        "mistralai/Mistral-7B-Instruct-v0.2": "mistral:instruct",
        "mistralai/Mistral-7B-Instruct-v0.1": "mistral:instruct",
        "meta-llama/Llama-2-7b-chat-hf": "llama2:7b-chat",
        "meta-llama/Llama-2-13b-chat-hf": "llama2:13b-chat",
    }
    
    return model_mapping.get(base_model_path, "mistral:instruct")


def get_model_parameters(model_dir: Path, adapter_config: dict) -> dict:
    """Determine appropriate model parameters based on configuration."""
    # Default parameters
    params = {
        "num_ctx": 2048,  # default context window
        "repeat_penalty": 1.15,  # helps reduce repetition
    }
    
    # Try to read training arguments to get max_length
    training_args_path = model_dir / "training_args.bin"
    if training_args_path.exists():
        try:
            import torch
            
            # Load with weights_only=False since this is our own training output
            # that we trust (not a downloaded checkpoint from untrusted source)
            training_args = torch.load(training_args_path, weights_only=False)
            
            # The max_length used during training should match num_ctx
            if hasattr(training_args, 'max_length'):
                params["num_ctx"] = training_args.max_length
        except Exception:
            print("⚠️  Warning: Could not read training_args.bin to determine max_length.")
            pass
    
    return params


def create_modelfile(
    model_dir: Path,
    output_path: Path,
    adapter_path: Optional[Path] = None,
) -> Path:
    """
    Create an Ollama Modelfile for the fine-tuned model.
    
    Args:
        model_dir: Directory containing the fine-tuned model
        output_path: Directory where the Modelfile should be created
        adapter_path: Optional explicit path to the adapter GGUF file
    
    Returns:
        Path to the created Modelfile
    """
    # Find adapter config
    config_path = find_adapter_config(model_dir)
    if not config_path:
        raise FileNotFoundError(
            f"adapter_config.json not found in {model_dir}. "
            "Make sure this is a valid LoRA adapter directory."
        )
    
    # Read model info
    model_info = read_base_model_info(config_path)
    base_model = model_info["base_model"]
    
    if not base_model:
        raise ValueError(
            f"Could not determine base model from {config_path}. "
            "Make sure adapter_config.json contains base model information."
        )
    
    # Determine Ollama base model name
    ollama_base = determine_base_model_name(base_model)
    
    # Get model parameters
    params = get_model_parameters(model_dir, model_info["config"])
    
    # Determine adapter path
    if adapter_path is None:
        # Look for the exported GGUF in the out/ directory
        out_dir = model_dir / "out"
        if out_dir.exists():
            # Find the lora adapter GGUF
            lora_files = list(out_dir.glob("lora_adaptor-*.gguf"))
            if lora_files:
                adapter_path = lora_files[0]
            else:
                raise FileNotFoundError(
                    f"No LoRA adapter GGUF found in {out_dir}. "
                    "Run export_lora.py first."
                )
        else:
            raise FileNotFoundError(
                f"Output directory {out_dir} not found. "
                "Run export_lora.py first to generate the GGUF adapter."
            )
    
    # Extract dataset type and timestamp from model_dir name
    # e.g., "gmail_2025-11-11_12-33" -> gmail
    dir_name = model_dir.name
    dataset_type = dir_name.split("_")[0]
    
    # Create Modelfile name
    username = model_dir.parent.name  # e.g., "alex" from models/alex/...
    modelfile_name = f"Modelfile.{username}_{dataset_type}"
    modelfile_path = output_path / modelfile_name
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create the Modelfile content
    modelfile_content = f"""FROM {ollama_base}
ADAPTER {adapter_path.resolve()}

PARAMETER num_ctx {params['num_ctx']}
PARAMETER repeat_penalty {params['repeat_penalty']}
"""
    
    # Write the Modelfile
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    print(f"✓ Created Modelfile: {modelfile_path}")
    print(f"\nModelfile contents:")
    print("=" * 60)
    print(modelfile_content)
    print("=" * 60)
    print(f"\nTo create the model in Ollama, run:")
    print(f"  cd {output_path}")
    print(f"  ollama create {username}_{dataset_type} -f {modelfile_name}")
    
    return modelfile_path


def main():
    parser = argparse.ArgumentParser(
        description="Create an Ollama Modelfile from a fine-tuned LoRA adapter"
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        help="Directory containing the fine-tuned model (e.g., models/alex/gmail_2025-11-11_12-33)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where the Modelfile should be created (e.g., ~/code/other/ollama-webui/models)",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional explicit path to the adapter GGUF file",
    )
    
    args = parser.parse_args()
    
    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.expanduser().resolve()
    
    if not model_dir.exists():
        print(f"Error: Model directory does not exist: {model_dir}")
        return 1
    
    try:
        create_modelfile(model_dir, output_dir, args.adapter)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
