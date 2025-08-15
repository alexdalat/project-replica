#!/usr/bin/env python3
"""
Export iMessage conversations from Apple Messages 'chat.db' to plain-text files.

- Creates ./imessages if it doesn't exist.
- One file per 1:1 chat (uses the other party's handle/phone number or contact name for the filename).
- One file per group chat (participants' names joined by commas, truncated if too long).
- Message format: 'DD/MM/YYYY, HH:MM - <Sender>: <Text>'
- Skips empty/non-text messages and common reaction/system lines.
- If message.text is empty, decodes message.attributedBody (typedstream).
- NEW: If a target filename already exists and its FIRST LINE matches this export's first line,
       the files are MERGED (no duplicate filenames).
"""

import argparse
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Iterable, List

from typedstream.stream import TypedStreamReader  # type: ignore

APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)

SYSTEM_SNIPPETS = (
    "This message responded to an earlier message.",
    "This message was deleted from the conversation!"
)


# --- Utility functions ---

def apple_time_to_dt(value: Optional[int]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        v = int(value)
    except Exception:
        return None
    if v > 10**15:  # ns
        seconds = v / 1_000_000_000
    elif v > 10**12:  # µs
        seconds = v / 1_000_000
    elif v > 10**10:  # ms
        seconds = v / 1_000
    else:  # s
        seconds = v
    dt_utc = APPLE_EPOCH + timedelta(seconds=seconds)
    return dt_utc.astimezone()



def load_contacts(path: Path) -> dict:
    if path and path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Contacts file not found: {path}")


def map_sender(sender: str, me_name: str, contacts: dict) -> str:
    s = sender.strip()
    if s == "Me":
        return me_name
    return contacts.get(s, s)


def sanitize_handle(handle: str) -> str:
    handle = handle.strip()
    phone_like = re.sub(r"[^\d+]", "", handle)
    if phone_like.startswith("+") and len(phone_like) > 5:
        return phone_like
    if re.fullmatch(r"\d{7,}", phone_like):
        return phone_like
    slug = re.sub(r"[^A-Za-z0-9._+-]", "_", handle)
    return slug or "unknown"


def normalize_text(text: str, ascii_punctuation: bool = True) -> str:
    from ftfy import fix_text
    import unicodedata

    s = "" if text is None else str(text)

    # Normalize newlines first
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Repair text (smart quotes gone wrong, mojibake, bad encodings, etc.)
    s = fix_text(s)

    # Keep accents/ligatures intact (é, œ, …): use NFC (NOT NFKC/NFKD)
    s = unicodedata.normalize("NFC", s)

    # Replace odd spaces commonly found in Apple/French typography with regular spaces
    for bad_space in ("\u00a0",  # no-break space
                      "\u202f",  # narrow no-break space
                      "\u2009"): # thin space
        s = s.replace(bad_space, " ")

    # Drop zero-width characters & soft hyphen
    for zw in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"):
        s = s.replace(zw, "")
    s = s.replace("\u00ad", "")  # soft hyphen

    # Optionally "defancy" punctuation -> ASCII, without touching letters/accents
    if ascii_punctuation:
        s = s.translate(str.maketrans({
            "\u2018": "'", "\u2019": "'", "\u201a": "'",
            "\u201c": '"', "\u201d": '"', "\u201e": '"',
            "\u2013": "-", "\u2014": "-", "\u2212": "-",
            "\u00b4": "'", "\u2032": "'", "\u2033": '"',
        }))

    # Strip other control characters but keep newlines and tabs
    s = "".join(ch for ch in s if ch in "\n\t" or not unicodedata.category(ch).startswith("C"))

    return s



def decode_attributed_body(data: Optional[bytes]) -> Optional[str]:
    """
    Decode Apple's typedstream 'attributedBody' into plain text.
    1) If pytypedstream is available, use it (preferred).
    2) Fallback: UTF-8 'ignore' + heuristics to recover visible text.
    """
    if not data:
        return None
    if isinstance(data, memoryview):
        data = bytes(data)

    # Preferred: pytypedstream (Apple 'typedstream' parser)
    try:
        for event in TypedStreamReader.from_data(data):
            if isinstance(event, bytes):
                return event.decode("utf-8", errors="replace")
            if isinstance(event, str):
                return event
    except Exception:
        pass  # fall back

    # Fallback heuristics
    s = data.decode("utf-8", errors="ignore")
    if "NSString" in s:
        chunk = s.split("NSString", 1)[1]
        if "NSDictionary" in chunk:
            chunk = chunk.split("NSDictionary", 1)[0]
        chunk = chunk.strip(' \t\r\n"[]{}():,;')
        if chunk:
            return chunk

    candidates = re.findall(r"[^\x00-\x1F\x7F]{5,}", s)
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0].strip()

    return None


def uniquify_path(p: Path) -> Path:
    if not p.exists():
        return p
    stem, suffix = p.stem, p.suffix
    i = 2
    while True:
        candidate = p.with_name(f"{stem} ({i}){suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def write_or_merge(out_file: Path, new_lines: List[str]) -> None:
    """
    If out_file exists and the first line matches new_lines[0], merge (dedupe, keep order).
    Otherwise, write to a unique filename (… (2).txt).
    """
    if not new_lines:
        return
    if out_file.exists():
        with out_file.open("r", encoding="utf-8") as f:
            existing_lines = [ln.rstrip("\n") for ln in f]
        if existing_lines and new_lines and existing_lines[0] == new_lines[0]:
            seen = set()
            merged: List[str] = []
            for ln in existing_lines + new_lines:
                if ln not in seen:
                    seen.add(ln)
                    merged.append(ln)
            with out_file.open("w", encoding="utf-8") as f:
                f.write("\n".join(merged) + "\n")
            print(f"Merged {len(new_lines):>5} lines -> {out_file}")
            return
        else:
            out_file = uniquify_path(out_file)
    with out_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
    print(f"Wrote  {len(new_lines):>5} lines -> {out_file}")


# --- Database query functions ---

def fetch_one_to_one_chats(conn: sqlite3.Connection, min_msgs=5):
    sql = """
    SELECT c.ROWID AS chat_id,
           h.id    AS handle_id_text,
           h.ROWID AS handle_rowid
    FROM chat c
    JOIN chat_handle_join chj ON chj.chat_id = c.ROWID
    JOIN handle h ON h.ROWID = chj.handle_id
    WHERE c.ROWID IN (
        SELECT chat_id
        FROM chat_handle_join
        GROUP BY chat_id
        HAVING COUNT(handle_id) = 1
    )
    AND (
        SELECT COUNT(*)
        FROM chat_message_join cmj
        JOIN message m ON m.ROWID = cmj.message_id
        WHERE cmj.chat_id = c.ROWID
          AND (
              (m.text IS NOT NULL AND TRIM(m.text) <> '')
              OR m.attributedBody IS NOT NULL
          )
    ) >= ?
    ORDER BY c.ROWID;
    """
    return conn.execute(sql, (min_msgs,)).fetchall()


def fetch_group_chats(conn: sqlite3.Connection, min_msgs=5):
    sql = """
    SELECT c.ROWID AS chat_id
    FROM chat c
    WHERE c.ROWID IN (
        SELECT chat_id
        FROM chat_handle_join
        GROUP BY chat_id
        HAVING COUNT(handle_id) > 1
    )
    AND (
        SELECT COUNT(*)
        FROM chat_message_join cmj
        JOIN message m ON m.ROWID = cmj.message_id
        WHERE cmj.chat_id = c.ROWID
          AND (
              (m.text IS NOT NULL AND TRIM(m.text) <> '')
              OR m.attributedBody IS NOT NULL
          )
    ) >= ?
    ORDER BY c.ROWID;
    """
    return conn.execute(sql, (min_msgs,)).fetchall()


def fetch_participants_for_chat(conn: sqlite3.Connection, chat_id: int) -> List[str]:
    sql = """
    SELECT h.id AS handle_id_text
    FROM chat_handle_join chj
    JOIN handle h ON h.ROWID = chj.handle_id
    WHERE chj.chat_id = ?
    ORDER BY h.id;
    """
    rows = conn.execute(sql, (chat_id,)).fetchall()
    return [r["handle_id_text"] for r in rows if r["handle_id_text"]]


def fetch_messages_for_chat(conn: sqlite3.Connection, chat_id: int):
    sql = """
    SELECT m.ROWID,
           m.text,
           m.attributedBody,
           m.associated_message_type,
           m.is_from_me,
           m.date,
           h.id AS sender_handle
    FROM message AS m
    JOIN chat_message_join AS cmj ON cmj.message_id = m.ROWID
    LEFT JOIN handle h ON h.ROWID = m.handle_id
    WHERE cmj.chat_id = ?
    ORDER BY m.date, m.ROWID;
    """
    return conn.execute(sql, (chat_id,)).fetchall()


# --- Helper(s) for filenames ---

def build_group_filename(handles: Iterable[str], contacts: dict, me_label: str) -> str:
    """
    Build a filename for a group chat based on participant names (mapped via contacts),
    sorted and joined with commas; truncated to a safe length.
    """
    parts = sorted({sanitize_handle(h) for h in handles if h})
    display = [map_sender(p, me_label, contacts) for p in parts]
    base = ", ".join(display).strip() or "group"
    base = re.sub(r"[\\/:*?\"<>|]+", "_", base)
    if len(base) > 120:
        short = ", ".join(display[:5])
        more = max(0, len(display) - 5)
        base = f"{short} +{more}"
    return base


# --- Main export function ---

def main():
    parser = argparse.ArgumentParser(
        description="Export Apple iMessage chat.db to plain text files."
    )
    parser.add_argument(
        "data",
        type=Path,
        help="Path to the data directory containing chat.db, contacts file, and output files (ex: data)",
    )
    parser.add_argument(
        "--me", default="Me", help='Label to use for your own messages (default: "Me")'
    )
    parser.add_argument(
        "--min-msgs",
        type=int,
        default=5,
        help="Minimum number of messages in a chat to export (default: 5)",
    )
    args = parser.parse_args()

    data_path = Path(args.data).expanduser()
    
    db_path = data_path / "chat.db"
    if not db_path.exists():
        raise SystemExit(f"chat.db not found at: {db_path}")
    
    contacts_path = data_path / "phone_to_name.json"
    if not contacts_path.exists():
        raise SystemExit(f"Contacts file not found at: {contacts_path}")
    contacts = load_contacts(contacts_path)

    out_dir = data_path / "imessages"
    out_dir.mkdir(parents=True, exist_ok=True)  # allow overwrite

    conn = sqlite3.connect(db_path, uri=True)
    conn.row_factory = sqlite3.Row

    try:
        # ---- 1:1 CHATS ----
        chats = fetch_one_to_one_chats(conn, args.min_msgs)
        print(f"Found {len(chats)} one-to-one chats with >= {args.min_msgs} messages.")

        for chat in chats:
            chat_id = chat["chat_id"]
            other_handle_text = chat["handle_id_text"] or "unknown"
            base_key = sanitize_handle(other_handle_text)
            base_name = map_sender(base_key, args.me, contacts)
            out_file = out_dir / f"{base_name}.txt"

            rows = fetch_messages_for_chat(conn, chat_id)
            lines: List[str] = []

            for r in rows:
                # if not 0, it's a reaction or attachment or something, skip
                if r["associated_message_type"] != 0:
                    continue

                text = r["text"]
                if not text or str(text).strip() == "":
                    text = decode_attributed_body(r["attributedBody"])

                if not text or str(text).isspace() or str(text) == "￼":
                    continue  # skip empty/attachment

                text = normalize_text(text)

                dt = apple_time_to_dt(r["date"])
                if dt is None:
                    continue

                timestamp = dt.strftime("%d/%m/%Y, %H:%M")
                sender = args.me if r["is_from_me"] == 1 else base_name
                msg = " ".join(str(text).splitlines()).strip()
                lines.append(f"{timestamp} - {sender}: {msg}")

            if lines:
                write_or_merge(out_file, lines)
            else:
                print(f"Skipped (no text messages): {base_name}")

        # ---- GROUP CHATS ----
        gchats = fetch_group_chats(conn, args.min_msgs)
        print(f"Found {len(gchats)} group chats with >= {args.min_msgs} messages.")

        for chat in gchats:
            chat_id = chat["chat_id"]
            participants = fetch_participants_for_chat(conn, chat_id)
            group_base = build_group_filename(participants, contacts, args.me)
            out_file = out_dir / f"{group_base}.txt"

            rows = fetch_messages_for_chat(conn, chat_id)
            lines: List[str] = []

            for r in rows:
                if r["associated_message_type"] != 0:
                    continue
                
                text = r["text"]
                if not text or str(text).strip() == "":
                    text = decode_attributed_body(r["attributedBody"])

                if not text or str(text).isspace() or str(text) == "￼":
                    continue

                text = normalize_text(text)

                dt = apple_time_to_dt(r["date"])
                if dt is None:
                    continue

                timestamp = dt.strftime("%d/%m/%Y, %H:%M")
                if r["is_from_me"] == 1:
                    sender = args.me
                else:
                    sender_handle = r["sender_handle"] or "unknown"
                    sender_key = sanitize_handle(sender_handle)
                    sender = map_sender(sender_key, args.me, contacts)

                msg = " ".join(str(text).splitlines()).strip()
                lines.append(f"{timestamp} - {sender}: {msg}")

            if lines:
                write_or_merge(out_file, lines)
            else:
                print(f"Skipped (no text messages): {group_base}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
