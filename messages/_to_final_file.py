import json
from pathlib import Path
from _load_msgs import preprocess_convo

# first run `to_formatted.py`

def main(input_dir: str, output_jsonl: str, name: str, role: str = "user"):
    folder_path = Path(input_dir)

    if not folder_path.is_dir():
        raise ValueError(f"Input directory {input_dir} does not exist or is not a directory.")
    
    # if folder has no txt files, raise error
    if not any(folder_path.glob("*.txt")):
        raise ValueError(f"No .txt files found in {input_dir}. Please run export_imessages.py first.")

    # Convert each chat txt in the folder
    for fp in folder_path.glob("*.txt"):
        file_output_path = folder_path / (fp.stem + ".jsonl")
        print(f"Converting {fp.name} -> {file_output_path.name}")
        preprocess_convo(fp, file_output_path, role=role, chat_owner=name)

    # Merge all per-file jsonl into one
    merged = []
    for fp in folder_path.glob("*.jsonl"):
        print(f"Merging {fp.name}")
        merged.append(fp.read_text())

    # new python
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for line in merged:
            f.write(line)
    print(f"Wrote {output_jsonl}")

if __name__ == "__main__":
    # parse args
    import argparse
    parser = argparse.ArgumentParser(description="Convert iMessage txt files to a single JSONL file.")
    # parser.add_argument("input_dir", type=str, help="Directory containing iMessage txt files.")
    # parser.add_argument("output_jsonl", type=str, help="Output JSONL file path.")
    parser.add_argument("data_dir", type=str, help="Directory containing 'imessages/' and to-be output file.")
    parser.add_argument("--name", type=str, default="Me", help='Label to use for your own messages (default: "Me")')
    parser.add_argument("--role", type=str, default="user", help="Role of the chat owner (default: 'user').")
    args = parser.parse_args()

    input_dir = Path(args.data_dir) / "imessages"
    output_jsonl = Path(args.data_dir) / "final.jsonl"

    main(input_dir, output_jsonl, args.name, args.role)