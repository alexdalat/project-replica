#!/usr/bin/env python3
"""
to_final_file.py

Convert exported Gmail threads (emails/*.json) into JSONL training records
for fine-tuning an LLM to write emails in your style.

Format: For each email exchange, create a training pair where:
- input: The email(s) received from others (context/prompt)
- output: Your response email

This teaches the model to respond to emails in your writing style.

Usage:
  python3 to_final_file.py data_alex [--output final_gmail.jsonl]

Output JSONL lines:
  {"input": "<incoming email context>", "output": "<your response>"}
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple


def tidy_text(s: str) -> str:
    """Normalize newlines and trim trailing whitespace."""
    return "\n".join(
        line.rstrip()
        for line in s.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip()


def format_email_message(msg: Dict, include_metadata: bool = True) -> str:
    """
    Format a single email message for training.
    
    Args:
        msg: Message dict with keys: from, subject, date, body
        include_metadata: Whether to include From/Subject headers
    
    Returns:
        Formatted email text
    """
    parts = []
    
    if include_metadata:
        if msg.get("from"):
            parts.append(f"From: {msg['from']}")
        if msg.get("subject"):
            parts.append(f"Subject: {msg['subject']}")
        if parts:
            parts.append("")  # blank line after headers
    
    if msg.get("body"):
        parts.append(msg["body"].strip())
    
    return "\n".join(parts)


def extract_training_pairs(thread: Dict, include_incoming_metadata: bool = False) -> List[Tuple[str, str]]:
    """
    Extract training pairs from a thread.
    
    For each message you sent, pair it with only the immediate preceding
    message(s) from others (not the entire thread history).
    
    Args:
        thread: Thread data with messages
        include_incoming_metadata: Whether to include From/Subject for incoming messages
    
    Returns:
        List of (input_context, output_response) tuples
    """
    messages = thread.get("messages", [])
    if len(messages) < 2:
        return []
    
    pairs = []
    recent_incoming = []
    
    for msg in messages:
        if msg.get("is_from_me"):
            # This is your response - create a training pair
            if recent_incoming:
                # Format only the recent incoming message(s) as context
                context_parts = []
                for ctx_msg in recent_incoming:
                    formatted = format_email_message(
                        ctx_msg, 
                        include_metadata=include_incoming_metadata
                    )
                    if formatted:
                        context_parts.append(formatted)
                
                # Format your response (never include metadata for your own messages)
                response = format_email_message(msg, include_metadata=False)
                
                if context_parts and response:
                    input_text = "\n\n---\n\n".join(context_parts)
                    pairs.append((input_text, response))
                
                # Clear recent incoming after creating the pair
                recent_incoming = []
        else:
            # Message from someone else - add to recent incoming context
            recent_incoming.append(msg)
    
    return pairs


def process_thread_file(file_path: Path, include_incoming_metadata: bool = False) -> List[Tuple[str, str]]:
    """
    Process a single thread JSON file and extract training pairs.
    
    Args:
        file_path: Path to thread JSON file
        include_incoming_metadata: Whether to include From/Subject for incoming messages
    
    Returns:
        List of (input, output) tuples
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            thread = json.load(f)
        
        return extract_training_pairs(thread, include_incoming_metadata)
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return []


def run_all(input_dir: Path, output_jsonl: Path, include_incoming_metadata: bool = False):
    """
    Process all thread JSON files and create JSONL training data.
    
    Args:
        input_dir: Directory containing emails/ folder
        output_jsonl: Output JSONL file path
        include_incoming_metadata: Whether to include From/Subject for incoming messages
    """
    emails_dir = input_dir / "emails"
    
    if not emails_dir.is_dir():
        raise ValueError(
            f"Directory {emails_dir} does not exist. "
            "Run export_gmail.py first to export your emails."
        )
    
    # Get all JSON files
    json_files = sorted(emails_dir.glob("thread_*.json"))
    
    if not json_files:
        raise ValueError(
            f"No thread_*.json files found in {emails_dir}. "
            "Run export_gmail.py first."
        )
    
    print(f"Processing {len(json_files)} thread files...")
    
    total_pairs = 0
    all_pairs = []
    
    for json_file in json_files:
        pairs = process_thread_file(json_file, include_incoming_metadata)
        all_pairs.extend(pairs)
        total_pairs += len(pairs)
    
    # Write to JSONL
    with output_jsonl.open("w", encoding="utf-8") as f_out:
        for input_text, output_text in all_pairs:
            # Clean up the text
            input_text = tidy_text(input_text)
            output_text = tidy_text(output_text)
            
            # Skip if either is empty
            if not input_text or not output_text:
                continue
            
            record = {
                "input": input_text,
                "output": output_text
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"Wrote {total_pairs} training pairs to {output_jsonl}")
    print(f"\nTraining data format:")
    print(f"  - Input: Email(s) you received (from others)")
    print(f"  - Output: Your response email")
    print(f"\nThis data can be used to fine-tune an LLM to write emails in your style.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert exported Gmail threads into JSONL training data."
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing emails/ folder and where final JSONL will be written.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="final_gmail.jsonl",
        help="Output JSONL filename (default: final_gmail.jsonl).",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include From/Subject headers for incoming emails (default: off).",
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_path = data_dir / args.output
    
    run_all(
        input_dir=data_dir, 
        output_jsonl=output_path,
        include_incoming_metadata=args.include_metadata
    )


if __name__ == "__main__":
    main()
