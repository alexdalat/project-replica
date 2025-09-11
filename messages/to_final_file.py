#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm.auto import tqdm

# -----------------------------
# Config
# -----------------------------
SAME_CONVO_THRESHOLD_SECONDS = 3600  # max gap between messages to stay in same conversation
SAME_USER_THRESHOLD_SECONDS = 600    # merge consecutive messages from the same sender within this window

# iMessage "txt" line format:
# Top-level: "DD/MM/YYYY, HH:MM - Name: Message"
# Thread reply (from exporter): "    DD/MM/YYYY, HH:MM - Name: Message"
LINE_RE = re.compile(
    r'^(?P<indent>\s*)(?P<d>\d{2}/\d{2}/\d{4}), (?P<t>\d{2}:\d{2}) - (?P<user>[^:]+): (?P<msg>.*)$'
)

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
                flush_pending()
                indent_ws = m.group("indent")
                d_str, t_str = m.group("d"), m.group("t")
                user, msg = m.group("user"), m.group("msg")
                indent_level = len(indent_ws.expandtabs(4)) // 4
                dt = datetime.strptime(f"{d_str} {t_str}", "%d/%m/%Y %H:%M")
                pending = (dt, user.strip(), msg, indent_level)
            else:
                if pending is not None:
                    dt, user, msg, indent = pending
                    pending = (dt, user, msg + "\n" + line, indent)
                else:
                    continue

    flush_pending()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.reset_index(drop=True)

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
            normal_rows.append(row.to_dict())
            i += 1

    normal_df = pd.DataFrame(normal_rows) if normal_rows else pd.DataFrame(columns=df.columns)
    if not normal_df.empty:
        normal_df = normal_df.sort_values("date").reset_index(drop=True)
    return thread_blocks, normal_df

# -----------------------------
# Helpers for conversation shaping
# -----------------------------
def is_group_chat(df: pd.DataFrame, chat_owner: str) -> bool:
    """True if the thread contains more than one non-owner participant (or none)."""
    if df.empty:
        return True
    others = set(df["username"].unique()) - {chat_owner}
    # Require exactly one other participant to qualify as a 1:1 conversation
    return len(others) != 1

def df_add_time_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["date_previous"] = df["date"].shift(1)
    df["time_delta"] = (df["date"] - df["date_previous"]).dt.total_seconds()
    return df

def df_to_msg_dicts(df: pd.DataFrame, chat_owner: str):
    """Map usernames -> roles and emit list of {user, message, date} dicts."""
    df = df.sort_values("date").reset_index(drop=True)
    df["role"] = df["username"].apply(lambda u: "assistant" if u == chat_owner else "user")
    msgs = [
        {"user": row["role"], "message": row["message"], "date": row["date"].isoformat()}
        for _, row in df.iterrows()
    ]
    return msgs

def enforce_user_assistant_alternation(msgs):
    """
    Ensure the sequence:
      - starts with 'user'
      - alternates strictly user/assistant/user/assistant
      - ends with 'assistant'
    Strategy:
      - drop leading assistants
      - merge consecutive same-role messages
      - drop trailing 'user' if any
    """
    if not msgs:
        return []

    # Drop leading assistants
    i = 0
    while i < len(msgs) and msgs[i]["user"] != "user":
        i += 1
    msgs = msgs[i:]
    if not msgs:
        return []

    # Merge consecutive same-role messages
    merged = [msgs[0].copy()]
    for m in msgs[1:]:
        if m["user"] == merged[-1]["user"]:
            merged[-1]["message"] = merged[-1]["message"] + "\n" + m["message"]
            merged[-1]["date"] = m["date"]  # keep latest timestamp for the merged bubble
        else:
            merged.append(m.copy())

    # Ensure last is assistant
    if merged and merged[-1]["user"] != "assistant":
        merged = merged[:-1]

    # Final sanity: enforce alternation strictly (by removing any residual duplicates, just in case)
    alt = []
    expect = "user"
    for m in merged:
        if m["user"] == expect:
            alt.append(m)
            expect = "assistant" if expect == "user" else "user"
        else:
            # If it doesn't match expected, merge into the last of the same role if exists; otherwise skip.
            if alt and alt[-1]["user"] == m["user"]:
                alt[-1]["message"] = alt[-1]["message"] + "\n" + m["message"]
                alt[-1]["date"] = m["date"]
            # else skip (shouldn't really happen after previous steps)

    # Trim to end with assistant
    if alt and alt[-1]["user"] != "assistant":
        alt = alt[:-1]

    # Require at least one full turn (user -> assistant)
    if len(alt) < 2:
        return []
    return alt

def build_conversations_from_df(
    df: pd.DataFrame,
    chat_owner: str,
    use_time_gaps: bool,
):
    """
    Convert a (pre-segmented) DataFrame into a list of conversations (arrays of dicts).
    """
    if df.empty:
        return []

    df2 = df_add_time_deltas(df)

    outputs = []

    if use_time_gaps:
        # Split by time gaps
        cur_rows = []
        for idx, row in df2.iterrows():
            gap = row["time_delta"] if pd.notna(row.get("time_delta")) else (SAME_CONVO_THRESHOLD_SECONDS + 1)
            starts_new = (len(cur_rows) > 0 and gap >= SAME_CONVO_THRESHOLD_SECONDS)
            if starts_new:
                if cur_rows:
                    seg_df = pd.DataFrame(cur_rows)
                    msgs = df_to_msg_dicts(seg_df, chat_owner)
                    convo = enforce_user_assistant_alternation(msgs)
                    if convo:
                        outputs.append(convo)
                cur_rows = [row.to_dict()]
            else:
                cur_rows.append(row.to_dict())

        # Flush last
        if cur_rows:
            seg_df = pd.DataFrame(cur_rows)
            msgs = df_to_msg_dicts(seg_df, chat_owner)
            convo = enforce_user_assistant_alternation(msgs)
            if convo:
                outputs.append(convo)

    else:
        # Treat entire block as one conversation
        msgs = df_to_msg_dicts(df2, chat_owner)
        convo = enforce_user_assistant_alternation(msgs)
        if convo:
            outputs.append(convo)

    return outputs

def process_thread_file(
    txt_path: Path,
    chat_owner: str = "Me",
) -> list:
    """
    Convert one thread .txt into a list of conversations (arrays of {user, message, date}).
    - Threads (indented blocks) become their own conversations.
    - Normal messages are chunked by time gaps.
    - Entire file is skipped if it's a group chat (anything other than exactly one non-owner).
    """
    df = parse_thread_txt(txt_path)
    if df.empty:
        return []

    # Skip group chats
    if is_group_chat(df, chat_owner):
        return []

    # Segment into thread blocks and normal messages
    thread_blocks, normal_df = segment_threads(df)

    outputs = []

    # Thread blocks -> token-agnostic, keep together
    for block in thread_blocks:
        outputs.extend(
            build_conversations_from_df(
                block, chat_owner, use_time_gaps=False
            )
        )

    # Normal messages -> split by time gaps
    if not normal_df.empty:
        outputs.extend(
            build_conversations_from_df(
                normal_df, chat_owner, use_time_gaps=True
            )
        )

    return outputs

# -----------------------------
# Orchestrator
# -----------------------------
def run_all(
    input_dir: Path,
    output_json: Path,
    chat_owner_name: str,
    debug: bool = False,
):
    if not input_dir.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory.")

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {input_dir}. Please run export_imessages.py first.")

    if debug:
        txt_files = txt_files[:10]

    all_conversations = []
    for fp in tqdm(txt_files, desc="Processing threads"):
        convos = process_thread_file(
            fp,
            chat_owner=chat_owner_name,
        )
        all_conversations.extend(convos)

    with output_json.open("w", encoding="utf-8") as f_out:
        json.dump(all_conversations, f_out, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_conversations)} conversations to {output_json}")

# -----------------------------
# CLI
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert iMessage txt files into a JSON array of 1:1 example conversations (user/assistant alternating, ending with assistant)."
    )
    parser.add_argument("data_dir", type=str, help="Directory containing 'imessages/' and where output JSON will be written.")
    parser.add_argument("--name", type=str, default="Me", help='Label used for your own messages (mapped to user=\"assistant\"). Default: \"Me\"')
    parser.add_argument("--output", type=str, default="final_imsg.json", help='Output JSON filename (default: final_imsg.json).')
    parser.add_argument("--debug", action="store_true", help="Only run on first 10 files.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    input_dir = data_dir / "imessages"
    output_path = data_dir / args.output

    run_all(
        input_dir=input_dir,
        output_json=output_path,
        chat_owner_name=args.name,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()
