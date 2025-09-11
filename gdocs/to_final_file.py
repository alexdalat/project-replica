#!/usr/bin/env python3
"""
Build JSONL training records from Google Docs exported as plain text files in gdocs/*.

For each document:
- Use the entire document text as a single training example.

Output JSONL lines look like:
{"input": "<full document text>"}
"""

import json
import argparse
from pathlib import Path


def tidy_text(s: str) -> str:
    """Normalize newlines and trim trailing whitespace blocks."""
    return "\n".join(
        line.rstrip()
        for line in s.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip()


def run_all(input_dir: Path, output_jsonl: Path):
    gdocs_dir = input_dir / "gdocs"
    if not gdocs_dir.is_dir():
        raise ValueError(f"Directory {gdocs_dir} does not exist or is not a directory.")

    files = [p for p in sorted(gdocs_dir.iterdir()) if p.is_file()]
    if not files:
        raise ValueError(f"No files found in {gdocs_dir}. Place your exported docs there.")

    total_written = 0
    with output_jsonl.open("w", encoding="utf-8") as f_out:
        for fp in files:
            try:
                content = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                content = fp.read_bytes().decode("utf-8", errors="ignore")

            content = tidy_text(content)
            if not content:
                continue

            record = {"input": content}
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_written += 1

    print(f"Wrote {total_written} samples to {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Google Doc text files (gdocs/*) into plain-text JSONL training records."
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing gdocs/ and where final JSONL will be written.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="final_gdocs.jsonl",
        help="Output JSONL filename (default: final_gdocs.jsonl).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = data_dir / args.output

    run_all(input_dir=data_dir, output_jsonl=output_path)


if __name__ == "__main__":
    main()
