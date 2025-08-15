# inference.py
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from peft import PeftModel, PeftConfig
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ------------------------------------------------------------
# Config / env
# ------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("inference")

# Required: where your LoRA lives (root dir that contains checkpoints/* or final adapter files)
# e.g., models/alex/imsg_08-09-25
LORA_ROOT = os.getenv("LORA_WEIGHTS")
if not LORA_ROOT:
    raise ValueError(
        "LORA_WEIGHTS environment variable must be set to the path of your LoRA weights."
    )

# Optional: choose a checkpoint ("500" or "checkpoint-500"). If empty, pick the highest step
# If the root contains final adapter files (adapter_model.safetensors), that will be used.
LORA_CHECKPOINT = os.getenv("LORA_CHECKPOINT", "").strip() or None

HF_TOKEN = os.getenv("HF_TOKEN", None)  # optional

# Optional system prompt. Leave as None to omit.
# BASE_SYSTEM_PROMPT = None

# Example:
BASE_SYSTEM_PROMPT = """you're Alex Dalat.
- name: Alex Dalat (he/him)
- a college student at the University of Michigan.
- lives in Birmingham, Michigan (America/Detroit time).
- enjoys programming, working out, and spending time with friends.
"""

# Prompt format for your model family
PROMPT_TEMPLATE = "<start_header_id>{role}<end_header_id>{message}<|eot_id|>"
EOT = "<|eot_id|>"
GENERATION_ROLE = os.getenv("GENERATION_ROLE", "assistant")
INPUT_MAX_LENGTH = 2048


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _find_checkpoint_dir(root: Path, requested: Optional[str]) -> Path:
    """
    Decide which directory actually holds the adapter weights we want to load.
    Priority:
      1) If 'requested' is provided:
         - if it looks like a number ('500'), use root/f"checkpoint-{requested}"
         - if it looks like 'checkpoint-500' or an absolute/relative path, resolve appropriately
      2) If the root has final adapter files, use root
      3) Else choose the highest numeric checkpoint under root/checkpoint-*
    """
    root = root.resolve()

    if requested:
        # numeric like "500"
        if re.fullmatch(r"\d+", requested):
            cand = root / f"checkpoint-{requested}"
            if cand.is_dir():
                return cand
            raise FileNotFoundError(f"Requested checkpoint not found: {cand}")

        # 'checkpoint-500' style
        if requested.startswith("checkpoint-"):
            cand = root / requested
            if cand.is_dir():
                return cand
            raise FileNotFoundError(f"Requested checkpoint not found: {cand}")

        # treat as path
        cand = Path(requested)
        if cand.is_dir():
            return cand.resolve()
        raise FileNotFoundError(f"Requested checkpoint path not found: {requested}")

    # Use root if final artifacts live here
    if (root / "adapter_model.safetensors").exists() and (
        root / "adapter_config.json"
    ).exists():
        return root

    # Otherwise pick the highest step under checkpoint-*
    checkpoints = []
    for d in root.glob("checkpoint-*"):
        if d.is_dir():
            m = re.search(r"checkpoint-(\d+)$", d.name)
            if m:
                checkpoints.append((int(m.group(1)), d))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {root}")
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    return checkpoints[0][1]


def _read_base_model_id(adapter_dir: Path) -> str:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_dir}")
    with cfg_path.open() as f:
        data = json.load(f)
    # PEFT saves this key
    base_key_options = ["base_model_name_or_path", "base_model_name"]
    for k in base_key_options:
        if k in data:
            return data[k]
    raise KeyError(
        f"base model name not found in {cfg_path} (looked for {base_key_options})"
    )


# ------------------------------------------------------------
# Load model + tokenizer (4-bit)
# ------------------------------------------------------------
log.info("Resolving LoRA directory...")
LORA_DIR = _find_checkpoint_dir(Path(LORA_ROOT), LORA_CHECKPOINT)
log.info(f"Using adapter directory: {LORA_DIR}")

BASE_MODEL_ID = _read_base_model_id(LORA_DIR)
log.info(f"Base model (from adapter_config.json): {BASE_MODEL_ID}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

log.info("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    trust_remote_code=True,
    token=HF_TOKEN,
    device_map="auto",
)

log.info("Loading tokenizer...")
eval_tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    add_bos_token=True,
    trust_remote_code=True,
    use_fast=True,
    token=HF_TOKEN,
)
# match your fine-tuning choice
eval_tokenizer.pad_token = eval_tokenizer.unk_token

log.info("Attaching LoRA...")
ft_model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
ft_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {device}")
ft_model.to(device)


# ------------------------------------------------------------
# Generation
# ------------------------------------------------------------
def _encoded_len(text: str) -> int:
    # no special tokens to make accounting stable
    return len(eval_tokenizer.encode(text, add_special_tokens=False))


def format_query(history: List[Tuple[str, str]], message: str) -> str:
    """Build prompt from history + new message."""
    parts: List[str] = []
    total_len = 0

    # Optional system prompt
    if BASE_SYSTEM_PROMPT:
        sys_msg = PROMPT_TEMPLATE.format(role="system", message=BASE_SYSTEM_PROMPT)
        total_len += _encoded_len(sys_msg)
        parts.append(sys_msg)

    # Pack previous turns (most recent last in the prompt, but we build by inserting before new msg)
    # We add older ones only while staying under INPUT_MAX_LENGTH
    for user_msg, generated_msg in reversed(history):
        sys_fmt = PROMPT_TEMPLATE.format(role=GENERATION_ROLE, message=generated_msg)
        usr_fmt = PROMPT_TEMPLATE.format(role="user", message=user_msg)

        if (
            total_len + _encoded_len(sys_fmt) + _encoded_len(usr_fmt)
            >= INPUT_MAX_LENGTH
        ):
            print(f"We hit max history length: {INPUT_MAX_LENGTH}")
            break
        
        parts.append(sys_fmt)
        total_len += _encoded_len(sys_fmt)

        parts.append(usr_fmt)
        total_len += _encoded_len(usr_fmt)
        
    # Reverse to have the most recent user message last
    parts.reverse()

    # New user message last
    new_msg = PROMPT_TEMPLATE.format(role="user", message=message)
    parts.append(new_msg)

    # Model should continue as assistant
    return "".join(parts) + f"<start_header_id>{GENERATION_ROLE}<end_header_id>"


def generate_with_model(
    eval_prompt, temperature, repetition_penalty, max_new_tokens
):
    model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to(device)
    input_len = model_input["input_ids"].shape[-1]

    with torch.no_grad():
        output = ft_model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            temperature=temperature,
            stop_strings=EOT.split(","),
            tokenizer=eval_tokenizer,
        )[0]

    # only the suffix (= newly generated tokens)
    gen_ids = output[input_len:]
    gen_text = eval_tokenizer.decode(gen_ids, skip_special_tokens=True)

    gen_text = gen_text.split("<start_header_id>{GENERATION_ROLE}<end_header_id>")[-1]
    gen_text = gen_text.split("<|eot_id|>")[0].strip()

    # remove in production
    print(f"input: {eval_prompt}")
    print(f"output: {gen_text}")

    return gen_text


# ------------------------------------------------------------
# FastAPI app
# ------------------------------------------------------------
app = FastAPI(title="Webchat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== SESSION STATE (in-memory) ======
# history: List[Tuple[user_msg, system_msg]]
chat_sessions: Dict[str, List[Tuple[str, str]]] = {}


# ====== Schemas ======


class WebchatRequest(BaseModel):
    client_id: str
    message: str
    temperature: float = 0.7
    max_new_tokens: int = 200
    repetition_penalty: float = 1.15


class WebchatResponse(BaseModel):
    reply: str
    history_len: int


@app.get("/healthz")
def healthz():
    return {
        "status": "ok",
        "device": str(device),
        "adapter_dir": str(LORA_DIR),
        "base_model": BASE_MODEL_ID,
    }


@app.post("/webchat/send", response_model=WebchatResponse)
async def webchat_send(req: WebchatRequest):
    history = chat_sessions.get(req.client_id, [])

    prompt = format_query(history, req.message)
    generated_text = generate_with_model(
        eval_prompt=prompt,
        temperature=req.temperature,
        repetition_penalty=req.repetition_penalty,
        max_new_tokens=req.max_new_tokens,
    )

    reply = generated_text

    # Update history
    history = history + [(req.message, reply)]
    chat_sessions[req.client_id] = history

    return WebchatResponse(reply=reply, history_len=len(history))


@app.post("/webchat/clear")
async def clear_history(client_id: str):
    chat_sessions.pop(client_id, None)
    return {"status": "ok"}


# Static UI (optional)
if Path("static/chat.html").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse("static/chat.html")
