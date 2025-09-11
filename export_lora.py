#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

POSSIBLE_KEYS = [
    "base_model_name_or_path",
    "base_model_name",
    "base_model",
]

def find_adapter_config(data_dir: Path) -> Path:
    candidates = list(data_dir.glob("adapter_config.json")) + list(data_dir.glob("**/adapter_config.json"))
    if not candidates:
        raise FileNotFoundError(f"adapter_config.json not found under {data_dir}")
    return sorted(candidates, key=lambda p: len(p.parts))[0]

def read_base_from_config(cfg_path: Path) -> str:
    data = json.loads(cfg_path.read_text())
    for k in POSSIBLE_KEYS:
        if k in data and data[k]:
            return data[k]
    raise KeyError(f"No base model key found in {cfg_path}")

def _clear_path(path: Path):
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

def _make_dir_link(link_path: Path, target: Path) -> Path:
    """
    Create a directory symlink (Unix/macOS) or junction (Windows). If not permitted,
    fall back to copying (last resort).
    """
    if link_path.exists() or link_path.is_symlink():
        # If already pointing to the right place, keep it
        try:
            if link_path.resolve() == target.resolve():
                return link_path
        except Exception:
            pass
        _clear_path(link_path)

    try:
        os.symlink(target, link_path, target_is_directory=True)
        print(f"[info] Symlinked {link_path} -> {target}")
        return link_path
    except Exception as e:
        if os.name == "nt":
            # Windows junction
            try:
                subprocess.check_call(["cmd", "/c", "mklink", "/J", str(link_path), str(target)])
                print(f"[info] Junction created {link_path} -> {target}")
                return link_path
            except Exception as e2:
                print(f"[warn] Could not create junction: {e2}")
        print(f"[warn] Symlink not permitted; copying instead (may take a while). Reason: {e}")
        shutil.copytree(target, link_path)
        return link_path

def resolve_or_link_base(base_id_or_path: str, base_dest_dir: Path) -> Path:
    """
    - If base_id_or_path is a local path -> link base_dest_dir -> that folder.
    - Else treat as HF repo id:
        * Try cache-only snapshot_download; if found, link base_dest_dir -> cache snapshot.
        * If not found, download into base_dest_dir.
    Returns the resolved local path that base_dest_dir points to (or contains).
    """
    base_dest_dir.parent.mkdir(parents=True, exist_ok=True)

    # Local path case
    local_path = Path(base_id_or_path)
    if local_path.exists():
        return _make_dir_link(base_dest_dir, local_path.resolve())

    # HF repo case: try local cache first
    try:
        cache_snapshot = snapshot_download(
            repo_id=base_id_or_path,
            local_files_only=True,
            local_dir_use_symlinks=False,
        )
        cache_path = Path(cache_snapshot).resolve()
        print(f"[info] Found base in HF cache: {cache_path}")
        return _make_dir_link(base_dest_dir, cache_path)
    except Exception as e:
        print(f"[info] Base not found in HF cache; will download to {base_dest_dir} ...")

    # Download into data_dir/base
    base_dest_dir.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(
        repo_id=base_id_or_path,
        local_dir=base_dest_dir,
        local_dir_use_symlinks=False,
    )
    return Path(local_dir).resolve()

def run(cmd: list):
    print("$", " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path, help="Folder containing adapter_config.json and adapter_model/")
    ap.add_argument("--llamacpp", type=Path, default=Path("./llama.cpp/"), help="Path to llama.cpp source folder (built)")
    ap.add_argument("--quant", default="q4_0", help="Quantization preset (e.g., q4_0, q5_0)")
    ap.add_argument("--full-export", action="store_true", help="Perform full export including base conversion, merging, and quantization")
    args = ap.parse_args()

    data_dir = args.data_dir.resolve()
    cfg_path = find_adapter_config(data_dir)
    base_model_or_path = read_base_from_config(cfg_path)
    print(f"[info] Detected base_model_name_or_path: {base_model_or_path}")

    base_link_dir = data_dir / "base"
    base_dir = resolve_or_link_base(base_model_or_path, base_link_dir)
    print(f"[info] Base model at (linked or downloaded): {base_dir}")

    outdir = data_dir / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    data_dir_name_part = "-".join(data_dir.parts[-2:])

    llamacpp = args.llamacpp.resolve()
    if not llamacpp.exists():
        raise FileNotFoundError(f"llama.cpp path does not exist: {llamacpp}")
    if not (llamacpp / "build").exists():
        raise FileNotFoundError(f"llama.cpp build directory not found: {llamacpp / 'build'}")

    base_f16_path = outdir / "base-f16.gguf"
    lora_fp16_path = outdir / "lora_adaptor-f16.gguf"

    merged_f16_path = outdir / f"merged-{data_dir_name_part}-f16.gguf"
    final_quant_path = outdir / f"merged-{data_dir_name_part}-{args.quant}.gguf"

    # 2) Convert LoRA to FP16 GGUF
    run([
        "python3",
        str(llamacpp / "convert_lora_to_gguf.py"),
        "--base", str(base_dir),
        "--outfile", str(lora_fp16_path),
        "--outtype", "f16",
        str(data_dir),
    ])

    if not args.full_export:
        print("[done] Converted LoRA to GGUF. Skipping merge/quant. Add --full-export to continue.")
        return
    
		# 1) Convert HF base to GGUF (F16)
    run([
        "python3",
        str(llamacpp / "convert_hf_to_gguf.py"),
        "--outfile", str(base_f16_path),
        str(base_dir),
    ])

    # 3) Merge LoRA into FP16 base
    run([
        str(llamacpp / "build" / "bin" / "llama-export-lora"),
        "-m", str(base_f16_path),
        "--lora", str(lora_fp16_path),
        "-o", str(merged_f16_path),
    ])

    # 4) Quantize merged model
    run([
        str(llamacpp / "build" / "bin" / "llama-quantize"),
        str(merged_f16_path),
        str(final_quant_path),
        "2" if args.quant.lower() == "q4_0" else args.quant.lower(),
    ])

		# 5) Delete intermediate files
    os.remove(base_f16_path)
    os.remove(lora_fp16_path)
    os.remove(merged_f16_path)
    print(f"[info] Cleaned up intermediate files.")

    print(f"[success] Final model: {final_quant_path}")

if __name__ == "__main__":
    main()
