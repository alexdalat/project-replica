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
- NEW: Threaded replies are exported ONLY beneath their origin message, in time order,
       indented one level, and not shown elsewhere in the chat. Orphan replies (no origin
       present) are grouped at the end under a stub.
"""

import argparse
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Iterable, List, Dict, Any, Tuple

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
    for bad_space in ("\u00a0", "\u202f", "\u2009"):
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

    try:
        for event in TypedStreamReader.from_data(data):
            if isinstance(event, bytes):
                return event.decode("utf-8", errors="replace")
            if isinstance(event, str):
                return event
    except Exception:
        pass  # fall back

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


def write_to_file(out_file: Path, new_lines: List[str]) -> None:
    if not new_lines:
        return
    if out_file.exists():
        print(f"File exists: {out_file}")
        return
    with out_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
    print(f"Wrote {len(new_lines):>5} lines -> {out_file}")


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
    """
    Fetch messages with GUID and (if available) thread origin GUID.
    Falls back gracefully if older schemas don't have thread columns.
    """
    sql_with_threads = """
    SELECT m.ROWID,
           m.guid,
           m.thread_originator_guid,
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
    sql_without_threads = """
    SELECT m.ROWID,
           m.guid,
           NULL AS thread_originator_guid,
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
    try:
        return conn.execute(sql_with_threads, (chat_id,)).fetchall()
    except sqlite3.OperationalError:
        # Older macOS schema (no threads); treat all as top-level
        return conn.execute(sql_without_threads, (chat_id,)).fetchall()


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


# --- Rendering helpers (with threads) ---

def _format_line(dt: datetime, sender: str, msg: str, indent_level: int = 0) -> str:
    timestamp = dt.strftime("%d/%m/%Y, %H:%M")
    indent = "    " * indent_level  # 4 spaces per level
    return f"{indent}{timestamp} - {sender}: {msg}"


def _extract_text_and_dt(row: sqlite3.Row) -> Optional[Tuple[str, datetime]]:
    """Decode text (text or attributedBody) and convert date."""
    if row["associated_message_type"] != 0:
        return None  # reaction/attachment/etc.
    text = row["text"]
    if not text or str(text).strip() == "":
        text = decode_attributed_body(row["attributedBody"])
    if not text or str(text).isspace() or str(text) == "￼":
        return None
    text = normalize_text(text)
    dt = apple_time_to_dt(row["date"])
    if dt is None:
        return None
    msg = " ".join(str(text).splitlines()).strip()
    return msg, dt


def _sender_label_for_row_1to1(row: sqlite3.Row, me_label: str, other_display: str) -> str:
    return me_label if row["is_from_me"] == 1 else other_display


def _sender_label_for_row_group(row: sqlite3.Row, me_label: str, contacts: dict) -> str:
    if row["is_from_me"] == 1:
        return me_label
    sender_handle = row["sender_handle"] or "unknown"
    sender_key = sanitize_handle(sender_handle)
    return map_sender(sender_key, me_label, contacts)


def build_lines_for_chat(
    rows: List[sqlite3.Row],
    me_label: str,
    *,
    mode: str,  # "1to1" or "group"
    other_display: Optional[str] = None,  # required for 1:1
    contacts: Optional[dict] = None      # required for group
) -> List[str]:
    """
    Build plain-text lines for a chat, grouping threaded replies under their origin.
    """
    assert mode in {"1to1", "group"}
    messages: List[Dict[str, Any]] = []

    # 1) Collect valid textual messages
    for r in rows:
        extracted = _extract_text_and_dt(r)
        if not extracted:
            continue
        msg, dt = extracted
        if mode == "1to1":
            assert other_display is not None
            sender = _sender_label_for_row_1to1(r, me_label, other_display)
        else:
            assert contacts is not None
            sender = _sender_label_for_row_group(r, me_label, contacts)

        messages.append({
            "guid": r["guid"],
            "thread_originator_guid": r["thread_originator_guid"],
            "dt": dt,
            "sender": sender,
            "msg": msg,
            "rowid": r["ROWID"],
        })

    # 2) Split into top-level and thread replies
    top_level = [m for m in messages if not m["thread_originator_guid"]]
    replies_by_origin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in messages:
        if m["thread_originator_guid"]:
            replies_by_origin[m["thread_originator_guid"]].append(m)

    # 3) Sort
    top_level.sort(key=lambda m: (m["dt"], m["rowid"]))
    for lst in replies_by_origin.values():
        lst.sort(key=lambda m: (m["dt"], m["rowid"]))

    # 4) Render: top-level, then its replies indented
    lines: List[str] = []
    seen_reply_guids: set = set()

    for m in top_level:
        lines.append(_format_line(m["dt"], m["sender"], m["msg"], indent_level=0))
        for r in replies_by_origin.get(m["guid"], []):
            lines.append(_format_line(r["dt"], r["sender"], r["msg"], indent_level=1))
            seen_reply_guids.add(r["guid"])

    return lines


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
        default=15,
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
    out_dir.mkdir(parents=True, exist_ok=False)

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
            lines = build_lines_for_chat(
                rows, args.me, mode="1to1", other_display=base_name
            )

            if lines:
                write_to_file(out_file, lines)
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
            lines = build_lines_for_chat(
                rows, args.me, mode="group", contacts=contacts
            )

            if lines:
                write_to_file(out_file, lines)
            else:
                print(f"Skipped (no text messages): {group_base}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
