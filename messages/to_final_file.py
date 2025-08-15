#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from dotenv import load_dotenv

# -----------------------------
# Config
# -----------------------------
SAME_CONVO_THRESHOLD_SECONDS = 3600  # max gap between messages to stay in same conversation
SAME_USER_THRESHOLD_SECONDS = 600    # merge consecutive messages from the same sender within this window
HISTORY_MAX_TOKENS = 2048            # max tokens for a single conversation block
CONVO_MIN_TOKENS = 100               # drop tiny conversations

# iMessage "txt" line format:
# Top-level: "DD/MM/YYYY, HH:MM - Name: Message"
# Thread reply (from exporter): "    DD/MM/YYYY, HH:MM - Name: Message"
LINE_RE = re.compile(
    r'^(?P<indent>\s*)(?P<d>\d{2}/\d{2}/\d{4}), (?P<t>\d{2}:\d{2}) - (?P<user>[^:]+): (?P<msg>.*)$'
)

# -----------------------------
# Tokenizer
# -----------------------------
def get_tokenizer():
    load_dotenv()
    base_model_id = os.getenv("base_model_id") or "alpindale/Mistral-7B-v0.2-hf"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_id,
            add_bos_token=False,
            trust_remote_code=True,
            use_fast=True,
            force_download=False,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    return tokenizer

# -----------------------------
# Parsing & Preprocessing
# -----------------------------
def parse_thread_txt(filepath: Path) -> pd.DataFrame:
    """
    Parse exported iMessage TXT into DataFrame with columns:
    - date (datetime)
    - username (str)
    - message (str)
    - indent (int)   # 0 for top-level; >0 for thread replies
    Supports multi-line messages.
    """
    rows = []
    pending = None  # (dt, user, msg_so_far, indent)

    def flush_pending():
        nonlocal pending
        if pending is not None:
            dt, user, msg, indent = pending
            rows.append({"date": dt, "username": user, "message": msg.strip(), "indent": indent})
            pending = None

    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = LINE_RE.match(line)
            if m:
                # flush previous pending
                flush_pending()
                # start new one
                indent_ws = m.group("indent")
                d_str, t_str = m.group("d"), m.group("t")
                user, msg = m.group("user"), m.group("msg")
                # treat any whitespace as indentation; exporter uses 4 spaces
                indent_level = len(indent_ws.expandtabs(4)) // 4
                dt = datetime.strptime(f"{d_str} {t_str}", "%d/%m/%Y %H:%M")
                pending = (dt, user.strip(), msg, indent_level)
            else:
                # continuation of previous message (multi-line) or non-dated stub -> append
                if pending is not None:
                    dt, user, msg, indent = pending
                    pending = (dt, user, msg + "\n" + line, indent)
                else:
                    # Line without a current message header; ignore safely.
                    # (Covers possible stub lines like "[replies to missing message ...]")
                    continue

    # flush last
    flush_pending()

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.reset_index(drop=True)

def collapse_same_sender(df: pd.DataFrame, delta_threshold: int = SAME_USER_THRESHOLD_SECONDS) -> pd.DataFrame:
    """
    Collapse consecutive rows from the same username when the gap between them
    is less than `delta_threshold` seconds. Recompute time deltas afterward.
    NOTE: Use this on already segmented data (threads or normal) so we don't merge across boundaries.
    """
    if df.empty:
        return df

    collapsed = []
    block = df.iloc[0].to_dict()  # current accumulating row

    for i in range(1, len(df)):
        cur = df.iloc[i]
        is_same_user = (cur["username"] == block["username"])
        dt_gap = (cur["date"] - df.iloc[i - 1]["date"]).total_seconds()
        if is_same_user and dt_gap < delta_threshold:
            block["message"] = f"{block['message']}\n{cur['message']}"
        else:
            collapsed.append(block)
            block = cur.to_dict()
    collapsed.append(block)

    out = pd.DataFrame(collapsed).sort_values("date").reset_index(drop=True)
    out["date_previous"] = out["date"].shift(1)
    out["time_delta"] = (out["date"] - out["date_previous"]).dt.total_seconds()
    return out

# -----------------------------
# Thread segmentation
# -----------------------------
def segment_threads(df: pd.DataFrame):
    """
    Split into:
      - thread_blocks: list of DataFrames, each starting with indent==0 followed by ≥1 rows indent>0
      - normal_df: DataFrame of top-level messages with no replies
    """
    thread_blocks = []
    normal_rows = []

    i = 0
    n = len(df)
    while i < n:
        row = df.iloc[i]
        if row["indent"] == 0:
            # collect following replies
            j = i + 1
            while j < n and df.iloc[j]["indent"] > 0:
                j += 1
            if j > i + 1:
                block = df.iloc[i:j].copy()
                thread_blocks.append(block)
            else:
                normal_rows.append(row.to_dict())
            i = j
        else:
            # A reply without a visible parent (rare). Treat as normal.
            normal_rows.append(row.to_dict())
            i += 1

    normal_df = pd.DataFrame(normal_rows) if normal_rows else pd.DataFrame(columns=df.columns)
    if not normal_df.empty:
        normal_df = normal_df.sort_values("date").reset_index(drop=True)
    return thread_blocks, normal_df

# -----------------------------
# Conversation assembly
# -----------------------------
def format_msg(role: str, content: str) -> str:
    # Match your original formatting
    return f"<start_header_id>{role}<end_header_id>{content}"

def chunk_msgs_by_tokens(msg_strs, tokenizer, max_tokens: int):
    """
    Greedy packer: yield joined conversations that do not exceed max_tokens.
    """
    out = []
    cur, cur_tokens = [], 0
    for s in msg_strs:
        n = len(tokenizer.encode(s))
        if cur and cur_tokens + n >= max_tokens:
            out.append("<|eot_id|>".join(cur) + "<|eot_id|>")
            cur, cur_tokens = [s], n
        else:
            cur.append(s)
            cur_tokens += n
    if cur:
        out.append("<|eot_id|>".join(cur) + "<|eot_id|>")
    return out

def rows_to_conversations(
    df: pd.DataFrame,
    tokenizer: AutoTokenizer,
    chat_owner: str,
    role_for_others: str,
    use_time_gaps: bool,
) -> list[str]:
    """
    Convert a (pre-segmented) DataFrame into conversations.
    If use_time_gaps=True: split by SAME_CONVO_THRESHOLD_SECONDS + token budget.
    If False: only split by token budget (used for a single thread block).
    """
    if df.empty:
        return []

    # Merge consecutive messages by the same sender within threshold
    df = collapse_same_sender(df, delta_threshold=SAME_USER_THRESHOLD_SECONDS)

    # Map to roles
    df["role"] = df["username"].apply(lambda u: "assistant" if u == chat_owner else role_for_others)

    outputs = []
    if use_time_gaps:
        # Original behavior: time-gap splits + token budget
        cur_msgs, cur_tokens = [], 0
        for _, row in df.iterrows():
            msg_str = format_msg(row["role"], row["message"])
            msg_len = len(tokenizer.encode(msg_str))

            gap = row["time_delta"] if pd.notna(row.get("time_delta", None)) else SAME_CONVO_THRESHOLD_SECONDS + 1
            starts_new_convo = (gap >= SAME_CONVO_THRESHOLD_SECONDS)
            exceeds_budget = (cur_tokens + msg_len >= HISTORY_MAX_TOKENS)

            if starts_new_convo or exceeds_budget:
                if cur_msgs:
                    query_str = "<|eot_id|>".join(cur_msgs) + "<|eot_id|>"
                    if len(tokenizer.encode(query_str)) > CONVO_MIN_TOKENS:
                        outputs.append(query_str)
                cur_msgs = [msg_str]
                cur_tokens = msg_len
            else:
                cur_msgs.append(msg_str)
                cur_tokens += msg_len

        if cur_msgs:
            query_str = "<|eot_id|>".join(cur_msgs) + "<|eot_id|>"
            if len(tokenizer.encode(query_str)) > CONVO_MIN_TOKENS:
                outputs.append(query_str)
    else:
        # Thread block: keep together; only enforce token budget (may split a very long thread)
        msg_strs = [format_msg("assistant" if (row["username"] == chat_owner) else role_for_others, row["message"])
                    for _, row in df.iterrows()]
        chunks = chunk_msgs_by_tokens(msg_strs, tokenizer, HISTORY_MAX_TOKENS)
        for ch in chunks:
            if len(tokenizer.encode(ch)) > CONVO_MIN_TOKENS:
                outputs.append(ch)

    return outputs

def process_thread_file(
    txt_path: Path,
    tokenizer: AutoTokenizer,
    chat_owner: str = "Me",
    role_for_others: str = "user",
) -> list[str]:
    """
    Convert one thread .txt into a list of conversation strings ready for JSONL,
    where each line is {"input": query_str}.

    Threads (indented blocks) become their own conversations.
    Normal (non-thread) messages are chunked with time-gaps + token budget (original behavior).
    """
    df = parse_thread_txt(txt_path)
    if df.empty:
        return []

    # Segment into thread blocks and normal messages
    thread_blocks, normal_df = segment_threads(df)

    outputs = []

    # 1) Threads -> each block to conversations (token budget only)
    for block in thread_blocks:
        outputs.extend(
            rows_to_conversations(
                block, tokenizer, chat_owner, role_for_others, use_time_gaps=False
            )
        )

    # 2) Normal -> original time-gap logic
    if not normal_df.empty:
        outputs.extend(
            rows_to_conversations(
                normal_df, tokenizer, chat_owner, role_for_others, use_time_gaps=True
            )
        )

    return outputs

# -----------------------------
# Orchestrator (no per-file intermediates)
# -----------------------------
def run_all(
    input_dir: Path,
    output_jsonl: Path,
    chat_owner_name: str,
    others_role: str = "user",
    debug: bool = False,
):
    if not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory.")

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {input_dir}. Please run export_imessages.py first.")

    if debug:
        txt_files = txt_files[:10]

    tokenizer = get_tokenizer()

    total_written = 0
    with output_jsonl.open("w", encoding="utf-8") as f_out:
        for fp in tqdm(txt_files, desc="Processing threads"):
            queries = process_thread_file(fp, tokenizer, chat_owner=chat_owner_name, role_for_others=others_role)
            for q in queries:
                f_out.write(json.dumps({"input": q}, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"Wrote {total_written} conversations to {output_jsonl}")

# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert iMessage txt files directly into a single JSONL (threads become their own conversations).")
    parser.add_argument("data_dir", type=str, help="Directory containing 'imessages/' and where final.jsonl will be written.")
    parser.add_argument("--name", type=str, default="Me", help='Label used for your own messages (mapped to role="assistant"). Default: "Me"')
    parser.add_argument("--role", type=str, default="user", help='Role for non-owner speakers (default: "user").')
    parser.add_argument("--output", type=str, default="final.jsonl", help='Output JSONL filename (default: final.jsonl).')
    parser.add_argument("--debug", action="store_true", help="Only run on first 10 files.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    input_dir = data_dir / "imessages"
    output_path = data_dir / args.output

    run_all(input_dir=input_dir, output_jsonl=output_path, chat_owner_name=args.name, others_role=args.role, debug=args.debug)

if __name__ == "__main__":
    main()
