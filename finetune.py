# finetune_lora.py
from __future__ import annotations

import math
import json
import os
import re
from pathlib import Path
from typing import Optional
from random import Random

from datetime import datetime
from datasets import load_dataset, Dataset
import torch

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
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
    # Test if wandb can actually connect
    try:
        wandb.init(mode="disabled")  # Test connection without actually logging
        wandb.finish()
        use_wandb = True
        print("✓ wandb connection successful")
    except Exception as e:
        use_wandb = False
        print(f"⚠ wandb available but can't connect: {e}")
        print("  → Falling back to tensorboard logging only")
except ImportError:
    use_wandb = False
    print("⚠ wandb not installed")
    print("  → Falling back to tensorboard logging only")

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

    if (root / "adapter_model.safetensors").exists() and (
        root / "adapter_config.json"
    ).exists():
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
    print(
        f"trainable params: {trainable} || all params: {total} || trainable%: {pct:.4f}"
    )


def add_len_column(ds):
    return ds.map(lambda x: {"_len": len(x["input_ids"])}, desc="Computing lengths")


def select_by_token_budget(ds, budget: int, seed: int = 42, desc: str = ""):
    # Shuffle deterministically, then take examples until we hit the budget
    ds = ds.shuffle(seed=seed)
    lengths = ds["_len"]
    selected_idx = []
    total = 0
    for i, L in enumerate(lengths):
        if total + L > budget:
            break
        selected_idx.append(i)
        total += L
    selected = ds.select(selected_idx)
    print(
        f"[{desc}] kept {len(selected)} examples, {total:,} tokens (budget={budget:,})."
    )
    return selected, total

def _apply_template_and_tokenize(example, tokenizer, args):
    """
    Convert [{"user": "...", "message": "..."}...] -> chat template text,
    then tokenize to get input_ids + attention_mask.

    Using apply_chat_template is required; we use tokenize=False first to get the
    formatted string, then call tokenizer(...) for masks/truncation control.
    """
    msgs = [
        {"role": m["user"], "content": m["message"]} for m in example["messages"]
    ]
    templated = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=False,  # SFT: no extra assistant turn
    )
    return tokenizer(
        templated,
        truncation=True,
        max_length=args.max_length,
        padding=False,  # let your DataCollator handle padding
    )

def _load_conversations_as_dataset(path: str) -> Dataset:
    """
    File format:
      [
        [ {"user":"user","message":"...","date":"..."}, {"user":"assistant","message":"...","date":"..."}, ... ],
        ...
      ]
    We store each conversation under a single 'messages' column.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array of conversations.")
    samples = [
        {"messages": conv}
        for conv in data
        if isinstance(conv, list) and len(conv) > 0
    ]
    return Dataset.from_list(samples)


# ------------------------- core -------------------------
def main(args):
    import os, math
    from pathlib import Path
    from datasets import load_dataset
    from transformers import TrainerCallback

    # use bf16 if neither bf16 nor fp16 is specified
    if not args.use_bf16 and not args.use_fp16:
        args.use_bf16 = True

    # ---------- figure out base model + (optional) init LoRA ----------
    hf_token = os.getenv("HF_TOKEN", None)

    base_model_id = args.base_model_id
    init_lora_dir = None

    if args.init_lora_dir:
        init_lora_dir = _find_checkpoint_dir(
            Path(args.init_lora_dir), args.lora_checkpoint
        )
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
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="left",
        add_eos_token=True,  # end of sentence token
        add_bos_token=True,  # beginning of sentence token
        token=hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ---------- datasets + tokenization ----------
    if args.train_dataset_path is None:
        raise ValueError("train_dataset_path must be provided.")
    if not args.train_dataset_path.exists():
        raise FileNotFoundError(f"Train dataset not found: {args.train_dataset_path}")
    if args.train_dataset_path.suffix not in (".json", ".jsonl"):
        raise ValueError(
            f"Unsupported train dataset format: {args.train_dataset_path.suffix}. "
            "Expected .json or .jsonl."
        )
    
    if args.train_dataset_path.suffix == ".json":  # assume it's conversation style
        train_dataset = _load_conversations_as_dataset(str(args.train_dataset_path))
        eval_dataset = (
            _load_conversations_as_dataset(str(args.eval_dataset_path))
            if args.eval_dataset_path
            else None
        )

        # If no eval provided, carve out a small validation split
        if eval_dataset is None:
            split = train_dataset.train_test_split(test_size=args.eval_fraction, seed=42)
            train_dataset, eval_dataset = split["train"], split["test"]
            print(
                f"No eval dataset provided, carving out {args.eval_fraction*100}% validation split from train dataset. "
                f"New sizes: train={len(train_dataset)}, eval={len(eval_dataset)}"
            )

        tokenized_train = train_dataset.map(
            _apply_template_and_tokenize, fn_kwargs={"tokenizer": tokenizer, "args": args}
        )
        tokenized_eval = eval_dataset.map(
            _apply_template_and_tokenize, fn_kwargs={"tokenizer": tokenizer, "args": args}
        )
    else:  # assume it's a single-column JSONL with "input" key
        train_dataset = load_dataset(
            "json",
            data_files=str(args.train_dataset_path),
            split="train",
            cache_dir="./cache",
        )

        eval_dataset = (
            load_dataset(
                "json",
                data_files=str(args.eval_dataset_path),
                split="train",
                cache_dir="./cache",
            )
            if args.eval_dataset_path
            else None
        )
        if eval_dataset is None:
            split = train_dataset.train_test_split(test_size=args.eval_fraction, seed=42)
            train_dataset, eval_dataset = split["train"], split["test"]
            print(
                f"No eval dataset provided, carving out {args.eval_fraction*100}% validation split from train dataset. "
                f"New sizes: train={len(train_dataset)}, eval={len(eval_dataset)}"
            )
        
        tokenized_train = train_dataset.map(
            lambda x: tokenizer(
                x["input"],
                truncation=True,
                max_length=args.max_length,
                padding=False,  # let your DataCollator handle padding
            ),
            desc="Tokenizing train dataset",
        )
        tokenized_eval = eval_dataset.map(
            lambda x: tokenizer(
                x["input"],
                truncation=True,
                max_length=args.max_length,
                padding=False,  # let your DataCollator handle padding
            ),
            desc="Tokenizing eval dataset",
        )

    # lengths
    tokenized_train = add_len_column(tokenized_train)
    tokenized_eval = add_len_column(tokenized_eval)

    # Optional: cap by token budget
    if args.token_budget is not None:
        if args.eval_dataset_path is None:
            eval_budget = max(1, int(args.token_budget * args.eval_fraction))
            train_budget = max(1, args.token_budget - eval_budget)
        else:
            eval_budget = max(1, int(args.token_budget * 0.1))
            train_budget = max(1, args.token_budget - eval_budget)

        tokenized_train, train_tokens = select_by_token_budget(
            tokenized_train, train_budget, args.seed, "train"
        )
        tokenized_eval, eval_tokens = select_by_token_budget(
            tokenized_eval, eval_budget, args.seed, "eval"
        )
        print(
            f"Combined token budget used: {train_tokens + eval_tokens:,} (train={train_tokens:,}, eval={eval_tokens:,})"
        )
    else:
        train_tokens = sum(tokenized_train["_len"])
        eval_tokens = sum(tokenized_eval["_len"])
        print(
            f"Token counts — train: {train_tokens:,}, eval: {eval_tokens:,}, total: {train_tokens + eval_tokens:,}"
        )

    # Preview a few examples
    i = 0
    if args.train_dataset_path.suffix == ".json":  # conversation style
        msgs = [{"role": m["user"], "content": m["message"]} for m in train_dataset[i]["messages"]]
        templated = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
    else:  # single-column JSONL
        templated = train_dataset[i]["input"]
    #tokenized = tokenized_train[i]["input_ids"]
    print(f"\n--- Preview of example #{i} ---\n{templated}")#\n{tokenized}")

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
            state_dict_config=FullStateDictConfig(
                offload_to_cpu=True, rank0_only=False
            ),
            optim_state_dict_config=FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=False
            ),
        )
        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        model = accelerator.prepare_model(model)

    # ---------- TrainingArguments with eval/best-model ----------
    evaluation_strategy = args.evaluation_strategy
    eval_steps = args.eval_steps
    eval_batch_size = args.batch_size

    # Output dir. ex with default output_dir: models/$USER/{dataset_type}_08-09-25_18-07
    # extract dataset type from train_dataset_path, e.g., "final_imsg.jsonl" -> "imsg", "final_gdoc.jsonl" -> "gdoc"
    dataset_type = args.train_dataset_path.stem.split("_")[-1]
    outdir = (
        args.output_dir / f"{dataset_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )

    # Determine logging backends based on wandb availability
    report_to = ["tensorboard"]
    if use_wandb:
        report_to.append("wandb")

    training_args = TrainingArguments(
        output_dir=outdir,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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
        report_to=report_to,
        run_name=f"{str(outdir).replace('/', '_')}",
        eval_on_start=True,
        max_grad_norm=0.3,
        eval_accumulation_steps=1,
    )

    es_callback = EarlyStoppingCallback(early_stopping_patience=2)

    print("Starting training...")
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

    p = argparse.ArgumentParser(
        description="Finetune a model with LoRA (QLoRA, 4-bit)."
    )

    # data
    p.add_argument(
        "train_dataset_path",
        type=Path,
        default=Path("data_" + os.getenv("USER", "unknown")) / "final_imsg.jsonl",
        help="Path to training dataset JSON (Default: 'data_$USER/final_imsg.jsonl').",
    )
    p.add_argument(
        "output_dir",
        type=Path,
        default=Path("models") / os.getenv("USER", "unknown"),
        help="Where to save LoRA checkpoints/adapters (Default: 'models/$USER/').",
    )
    p.add_argument(
        "--eval_dataset_path",
        type=Path,
        default=None,
        help="Optional eval/val dataset JSON.",
    )
    p.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length for training (Default: 1024).",
    )

    p.add_argument(
        "--token_budget",
        type=int,
        default=None,
        help="Total tokens to use across train+eval.",
    )
    p.add_argument(
        "--eval_fraction",
        type=float,
        default=0.05,
        help="If no eval set is provided, fraction of budget for eval.",
    )
    p.add_argument("--seed", type=int, default=42)

    # model selection
    p.add_argument(
        "--base_model_id", type=str, default=None, help="HF model id or local path."
    )
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
    p.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (α), weights scaling factor",
    )
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    p.add_argument(
        "--big_lora",
        action="store_true",
        help="Use bigger LoRA with additional modules.",
    )

    # training
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=2.5e-5)
    p.add_argument("--warmup_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=10, help="Log every N steps.")
    p.add_argument(
        "--logging_dir",
        type=str,
        default="./logs",
        help="Directory for training logs (tensorboard).",
    )
    p.add_argument(
        "--save_strategy", type=str, default="steps", choices=["epoch", "steps"]
    )
    p.add_argument("--save_steps", type=int, default=10)

    # evaluation
    p.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        choices=["no", "steps", "epoch"],
    )
    p.add_argument("--eval_steps", type=int, default=10)

    # precision
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--use_fp16", action="store_true")

    # resume
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Trainer checkpoint directory to resume.",
    )

    # optional FSDP
    p.add_argument(
        "--fsdp", type=bool, default=True, help="Wrap model with FSDP via Accelerate."
    )

    args = p.parse_args()
    main(args)
