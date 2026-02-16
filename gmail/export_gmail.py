#!/usr/bin/env python3
"""
export_gmail.py

Exports Gmail conversations as JSON files where each file represents an email thread.
Captures the context needed for training: incoming emails (user) and your responses (assistant).

Usage:
  python3 export_gmail.py data_alex [--years 5]

Environment (.env or process env):
  GOOGLE_CLIENT_ID       = <OAuth client id>
  GOOGLE_CLIENT_SECRET   = <OAuth client secret>
  GOOGLE_TOKEN_JSON      = <optional: full token json string for Credentials.from_authorized_user_info>

Output structure:
  <BASE_DIR>/emails/
    thread_<id>.json  - Each file contains a thread with messages
"""

import argparse
import base64
import email
import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from email.utils import parseaddr
from typing import Dict, List, Optional, Set

from tqdm import tqdm

# optional .env loader
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def build_gmail_service():
    """Build and return the Gmail API service with authentication."""
    creds = None
    token_path = "token.json"
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            # creds = flow.run_local_server(host="127.0.0.1", port=53682, open_browser=False, prompt="consent")
            # Use out-of-band flow for remote/Tailscale setups (no callback server needed)
            flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
            auth_url, _ = flow.authorization_url(prompt="consent")

            print("\n" + "=" * 70)
            print("AUTHENTICATION REQUIRED")
            print("=" * 70)
            print("\n1. Open this URL in your browser:\n")
            print(f"   {auth_url}\n")
            print("2. Authorize the application")
            print("3. Copy the authorization code from the browser\n")
            print("=" * 70)

            code = input("Enter the authorization code: ").strip()
            flow.fetch_token(code=code)
            creds = flow.credentials

        with open(token_path, "w", encoding="utf-8") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def get_user_email(service) -> str:
    """Get the authenticated user's email address."""
    profile = service.users().getProfile(userId="me").execute()
    return profile["emailAddress"]


def list_threads(
    service, created_after_iso: Optional[str] = None, max_retries: int = 5
) -> List[str]:
    """
    List all thread IDs where the user sent at least one message.
    Returns a list of thread IDs.
    """
    query = "in:sent -in:chats -in:spam -in:trash"
    if created_after_iso:
        # Convert ISO to Gmail date format (YYYY/MM/DD)
        dt = datetime.fromisoformat(created_after_iso.replace("Z", "+00:00"))
        date_str = dt.strftime("%Y/%m/%d")
        query += f" after:{date_str}"

    thread_ids = []
    page_token = None

    while True:
        attempt = 0
        while True:
            try:
                response = (
                    service.users()
                    .threads()
                    .list(
                        userId="me",
                        q=query,
                        maxResults=500,
                        pageToken=page_token,
                    )
                    .execute()
                )
                break
            except HttpError as e:
                status = getattr(e, "status_code", None) or (
                    e.resp.status if hasattr(e, "resp") else None
                )
                attempt += 1
                if attempt > max_retries or not status or status < 500:
                    raise
                sleep_s = min(2**attempt, 30)
                time.sleep(sleep_s)

        threads = response.get("threads", [])
        thread_ids.extend([t["id"] for t in threads])

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return thread_ids


def get_header(headers: List[Dict], name: str) -> str:
    """Extract a header value by name."""
    for h in headers:
        if h["name"].lower() == name.lower():
            return h["value"]
    return ""


def decode_body(part: Dict) -> str:
    """Decode the body of an email part."""
    if "data" in part.get("body", {}):
        data = part["body"]["data"]
        decoded = base64.urlsafe_b64decode(data).decode(
            "utf-8", errors="ignore"
        )
        return decoded
    return ""


def extract_text_from_payload(payload: Dict) -> str:
    """
    Extract plain text from email payload.
    Tries to get text/plain parts, falls back to text/html if needed.
    """
    mime_type = payload.get("mimeType", "")

    # Single part message
    if mime_type.startswith("text/plain"):
        return decode_body(payload)

    # Multipart message
    if "parts" in payload:
        plain_text = ""
        html_text = ""

        for part in payload["parts"]:
            part_mime = part.get("mimeType", "")

            if part_mime == "text/plain":
                plain_text += decode_body(part)
            elif part_mime == "text/html" and not plain_text:
                html_text += decode_body(part)
            elif "parts" in part:  # Nested multipart
                plain_text += extract_text_from_payload(part)

        # Prefer plain text, fall back to HTML
        text = plain_text or html_text

        # Basic HTML stripping if we only have HTML
        if not plain_text and html_text:
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"&nbsp;", " ", text)
            text = re.sub(r"&[a-z]+;", "", text)

        return text

    return ""


def clean_email_body(text: str) -> str:
    """
    Clean email body by removing quoted replies, signatures, etc.
    Keeps only the new content from this message.
    """
    # Remove inline quoted replies - multiple patterns:
    # 1. "On ... wrote:" (Gmail/standard format)
    # 2. "From: ... Sent: ..." (Outlook format)
    # 3. Other common quote patterns

    # Pattern 1: On ... wrote: (catches both middle and end of text)
    quote_pattern = r"On .+? wrote:.*"
    text = re.split(quote_pattern, text, maxsplit=1, flags=re.DOTALL)[0]

    # Pattern 2: From: ... (Outlook-style quoted messages)
    # Look for "From:" followed by email/name, then "Sent:" or "Subject:"
    outlook_pattern = r"\n\s*From:\s+.+?<[^>]+>.*"
    text = re.split(outlook_pattern, text, maxsplit=1, flags=re.DOTALL)[0]

    # Pattern 3: French-style Gmail mobile signatures
    # "Le {date} à {time}, {name} <email> a écrit :"
    french_pattern = r"Le .+? a écrit\s*:.*"
    text = re.split(french_pattern, text, maxsplit=1, flags=re.DOTALL)[0]

    # TODO: these are slow, optimize somehow

    lines = text.split("\n")
    result = []

    for line in lines:
        # Stop at common quote markers
        if line.strip().startswith(">"):
            break
        if line.strip() == "":
            result.append(line)
            continue

        # Skip email footers/signatures (common patterns)
        lower = line.lower()
        if any(
            marker in lower
            for marker in [
                "-- ",
                "sent from my",
                "get outlook for",
                "confidential",
                "disclaimer:",
            ]
        ):
            break

        result.append(line)

    # Join and clean up
    cleaned = "\n".join(result).strip()
    # Remove excessive whitespace
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def get_thread_messages(
    service, thread_id: str, user_email: str, max_retries: int = 5
) -> Optional[Dict]:
    """
    Fetch a thread and extract messages with metadata.
    Returns a dict with thread info and messages, or None if error/irrelevant.
    """
    attempt = 0
    while True:
        try:
            thread = (
                service.users()
                .threads()
                .get(userId="me", id=thread_id, format="full")
                .execute()
            )
            break
        except HttpError as e:
            status = getattr(e, "status_code", None) or (
                e.resp.status if hasattr(e, "resp") else None
            )
            attempt += 1
            if attempt > max_retries or not status or status < 500:
                return None
            sleep_s = min(2**attempt, 30)
            time.sleep(sleep_s)

    messages = []
    for msg in thread.get("messages", []):
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])

        subject = get_header(headers, "Subject")
        from_email = get_header(headers, "From")
        to_email = get_header(headers, "To")
        date = get_header(headers, "Date")

        # Parse sender email
        _, sender_addr = parseaddr(from_email)
        is_from_me = sender_addr.lower() == user_email.lower()

        # Extract body
        body = extract_text_from_payload(payload)
        body = clean_email_body(body)

        if not body.strip():
            continue

        messages.append(
            {
                "id": msg["id"],
                "date": date,
                "subject": subject,
                "from": from_email,
                "to": to_email,
                "is_from_me": is_from_me,
                "body": body,
            }
        )

    if not messages:
        return None

    return {
        "thread_id": thread_id,
        "subject": messages[0]["subject"],
        "messages": messages,
    }


def export_threads(
    service, thread_ids: List[str], output_dir: str, user_email: str
):
    """Export threads to JSON files."""
    os.makedirs(output_dir, exist_ok=True)

    exported = 0
    skipped = 0
    failed = 0

    for thread_id in tqdm(
        thread_ids, desc="Exporting Gmail threads", unit="thread"
    ):
        try:
            thread_data = get_thread_messages(service, thread_id, user_email)

            if not thread_data:
                skipped += 1
                continue

            # Write to file
            output_path = os.path.join(output_dir, f"thread_{thread_id}.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(thread_data, f, ensure_ascii=False, indent=2)

            exported += 1

        except Exception as e:
            print(f"Failed to export thread {thread_id}: {e}", file=sys.stderr)
            failed += 1

    return exported, skipped, failed


def main():
    parser = argparse.ArgumentParser(
        description="Export Gmail threads to JSON files for LLM training."
    )
    parser.add_argument(
        "base_dir", help="Base directory to create the 'emails' folder in."
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Export only emails from the last N years (default: 5).",
    )
    args = parser.parse_args()

    output_dir = os.path.join(args.base_dir, "emails")

    # Calculate cutoff date
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.years * 365)
    cutoff_iso = (
        cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    print("Building Gmail service...")
    service = build_gmail_service()

    print("Getting user email...")
    user_email = get_user_email(service)
    print(f"Authenticated as: {user_email}")

    print("Fetching thread list...")
    thread_ids = list_threads(service, created_after_iso=cutoff_iso)
    print(f"Found {len(thread_ids)} threads")

    if not thread_ids:
        print("No threads found. Exiting.")
        return

    print("Exporting threads...")
    exported, skipped, failed = export_threads(
        service, thread_ids, output_dir, user_email
    )

    print("\nExport complete!")
    print(f"  Exported: {exported}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
