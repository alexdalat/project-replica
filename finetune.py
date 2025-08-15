# finetune_lora.py
from __future__ import annotations

import math
import json
import os
import re
from pathlib import Path
from typing import Optional

from datetime import datetime
import torch
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
from datasets import load_dataset
from dotenv import load_dotenv
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
# Optional FSDP bits
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)

try:
    import wandb
    use_wandb = True
except ImportError:
    use_wandb = False

load_dotenv()


# ------------------------- helpers -------------------------
def _find_checkpoint_dir(root: Path, requested: Optional[str]) -> Path:
    """
    Choose which adapter directory to use.
    Priority:
      1) If `requested` is provided:
           - "500" -> root/checkpoint-500
           - "checkpoint-500" -> root/checkpoint-500
           - an explicit path -> that path
      2) If root has final adapter files, use root
      3) Else pick the highest checkpoint under root/checkpoint-*
    """
    root = root.resolve()

    if requested:
        if re.fullmatch(r"\d+", requested):
            cand = root / f"checkpoint-{requested}"
            if cand.is_dir():
                return cand
            raise FileNotFoundError(f"Requested checkpoint not found: {cand}")
        if requested.startswith("checkpoint-"):
            cand = root / requested
            if cand.is_dir():
                return cand
            raise FileNotFoundError(f"Requested checkpoint not found: {cand}")
        cand = Path(requested)
        if cand.is_dir():
            return cand.resolve()
        raise FileNotFoundError(f"Requested checkpoint path not found: {requested}")

    if (root / "adapter_model.safetensors").exists() and (root / "adapter_config.json").exists():
        return root

    candidates = []
    for d in root.glob("checkpoint-*"):
        m = re.search(r"checkpoint-(\d+)$", d.name)
        if d.is_dir() and m:
            candidates.append((int(m.group(1)), d))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {root}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _read_base_model_id(adapter_dir: Path) -> str:
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    data = json.loads(cfg.read_text())
    for k in ("base_model_name_or_path", "base_model_name"):
        if k in data:
            return data[k]
    raise KeyError(f"base model name not found in {cfg}")


def print_trainable_parameters(model):
    trainable, total = 0, 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    pct = 100 * trainable / total if total else 0
    print(f"trainable params: {trainable} || all params: {total} || trainable%: {pct:.4f}")


# ------------------------- core -------------------------
def main(args):
    import os, math
    from pathlib import Path
    from datasets import load_dataset
    from transformers import TrainerCallback

    # use bf16 if neither bf16 nor fp16 is specified
    if not args.use_bf16 and not args.use_fp16:
        args.use_bf16 = True

    # ---------- dataset ----------
    train_dataset = load_dataset("json", data_files=str(args.train_dataset_path), split="train")
    eval_dataset = (
        load_dataset("json", data_files=str(args.eval_dataset_path), split="train")
        if args.eval_dataset_path
        else None
    )

    if args.dataset_limit is not None:
        limit = min(args.dataset_limit, len(train_dataset))
        train_dataset = train_dataset.shuffle(seed=42).select(range(limit))
        if eval_dataset is not None:
            elimit = min(args.dataset_limit, len(eval_dataset))
            eval_dataset = eval_dataset.shuffle(seed=42).select(range(elimit))

    # If no eval provided, carve out a small validation split
    if eval_dataset is None:
        split = train_dataset.train_test_split(test_size=0.05, seed=42)
        train_dataset, eval_dataset = split["train"], split["test"]
        print(f"No eval dataset provided, carving out 5% validation split from train dataset. New sizes: "
              f"train={len(train_dataset)}, eval={len(eval_dataset)}")

    # ---------- figure out base model + (optional) init LoRA ----------
    hf_token = os.getenv("HF_TOKEN", None)

    base_model_id = args.base_model_id
    init_lora_dir = None

    if args.init_lora_dir:
        init_lora_dir = _find_checkpoint_dir(Path(args.init_lora_dir), args.lora_checkpoint)
        base_model_id = base_model_id or _read_base_model_id(init_lora_dir)

    if not base_model_id:
        base_model_id = os.getenv("BASE_MODEL_ID") or os.getenv("base_model_id")
    if not base_model_id:
        base_model_id = "mistralai/Mistral-7B-Instruct-v0.3"
        print(f"No base_model_id provided, using default: {base_model_id}")

    # ---------- model + tokenizer (QLoRA 4-bit) ----------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,
        add_bos_token=True,
        token=hf_token,
    )
    # Use EOS as pad; collator will mask pads out of loss
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_row(example):
        # Let the DataCollator create labels and ignore pads; no fixed-length padding here
        return tokenizer(
            example["input"],
            truncation=True,
            max_length=args.max_length,
            #padding="max_length",
        )

    tokenized_train = train_dataset.map(tokenize_row)
    tokenized_eval = eval_dataset.map(tokenize_row)

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # ---------- LoRA: either resume existing adapter or create new one ----------
    if init_lora_dir:
        model = PeftModel.from_pretrained(model, str(init_lora_dir))
        print("Loaded existing LoRA from:", init_lora_dir)
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if args.big_lora:
            target_modules.extend(["gate_proj", "up_proj", "down_proj", "lm_head"])

        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            bias="none",
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)

    print_trainable_parameters(model)

    # (Optional) FSDP wrapping
    if args.fsdp:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )
        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        model = accelerator.prepare_model(model)

    # ---------- TrainingArguments with eval/best-model ----------
    evaluation_strategy = args.evaluation_strategy
    eval_steps = args.eval_steps
    eval_batch_size = args.batch_size

    # Output dir. ex with default output_dir: models/$USER/imsg_08-09-25_18-07
    # imsg is hardcoded for now, but will be changed later when more datasets are supported
    outdir = args.output_dir / f"imsg_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"

    training_args = TrainingArguments(
        output_dir=outdir,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        bf16=args.use_bf16,
        fp16=args.use_fp16,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        save_strategy=args.save_strategy,  # "epoch" or "steps"
        save_steps=args.save_steps,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps if evaluation_strategy == "steps" else None,
        per_device_eval_batch_size=eval_batch_size,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=4,
        group_by_length=True,
        report_to="all",
        run_name=f"{str(outdir).replace('/', '_')}",
    )
    
    es_callback = EarlyStoppingCallback(early_stopping_patience=2)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[es_callback] if evaluation_strategy != "no" else [],
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)



if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Finetune a model with LoRA (QLoRA, 4-bit).")

    # data
    p.add_argument("train_dataset_path", type=Path, default=Path("data_" + os.getenv("USER", "unknown")), help="Path to training dataset JSON (Default: 'data_$USER/').")
    p.add_argument("output_dir", type=Path, default=Path("models") / os.getenv("USER", "unknown"), help="Where to save LoRA checkpoints/adapters (Default: 'models/$USER/').")
    p.add_argument("--eval_dataset_path", type=Path, default=None, help="Optional eval/val dataset JSON.")
    p.add_argument("--dataset_limit", type=int, default=None, help="Limit number of examples for quick runs.")
    p.add_argument("--max_length", type=int, default=2048)

    # model selection
    p.add_argument("--base_model_id", type=str, default=None, help="HF model id or local path.")
    p.add_argument(
        "--init-lora-dir",
        type=str,
        default=None,
        help="Resume from an existing LoRA directory (e.g., models/alex/imsg_08-09-25).",
    )
    p.add_argument(
        "--lora-checkpoint",
        type=str,
        default=None,
        help='Pick a specific checkpoint under init-lora-dir (e.g., "500" or "checkpoint-500").',
    )

    # LoRA hyperparams (used only when NOT resuming an existing adapter)
    # for r and alpha: https://datascience.stackexchange.com/questions/123229/understanding-alpha-parameter-tuning-in-lora-paper
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank (r)")
    p.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha (α), weights scaling factor") 
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    p.add_argument("--big_lora", action="store_true", help="Use bigger LoRA with additional modules.")

    # training
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2.5e-5)
    p.add_argument("--warmup_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=10, help="Log every N steps.")
    p.add_argument("--logging_dir", type=str, default="./logs", help="Directory for training logs (tensorboard).")
    p.add_argument("--save_strategy", type=str, default="steps", choices=["epoch", "steps"])
    p.add_argument("--save_steps", type=int, default=10)
    
    # evaluation
    p.add_argument("--evaluation_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    p.add_argument("--eval_steps", type=int, default=10)

    # precision
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--use_fp16", action="store_true")

    # resume
    p.add_argument("--resume_from_checkpoint", type=str, default=None, help="Trainer checkpoint directory to resume.")

    # optional FSDP
    p.add_argument("--fsdp", type=bool, default=True, help="Wrap model with FSDP via Accelerate.")

    args = p.parse_args()
    main(args)
