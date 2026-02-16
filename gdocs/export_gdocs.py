#!/usr/bin/env python3
"""
export_gdocs.py

Reads all Google Docs in your Google Drive and writes them to a local folder
<BASE_DIR>/gdocs where each file is named after the doc's title and contains
the document's plaintext.

Usage:
  python3 export_gdocs.py data_alex [--exclude_formatting]

Environment (.env or process env):
  GOOGLE_CLIENT_ID       = <OAuth client id>
  GOOGLE_CLIENT_SECRET   = <OAuth client secret>
  GOOGLE_TOKEN_JSON      = <optional: full token json string for Credentials.from_authorized_user_info>

If GOOGLE_CLIENT_ID/GOOGLE_CLIENT_SECRET are set, the OAuth flow uses them
instead of credentials.json. If GOOGLE_TOKEN_JSON is set, it is used instead of token.json.
"""

import argparse
import io
import re
import os
import re
import sys
import time
import json
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm
from datetime import datetime, timezone, timedelta

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

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
MAX_BASENAME_LEN = 180  # leave room for extension and potential suffix



MAX_URL_CHARS = 80  # adjust as needed

_MD_IMAGE_INLINE = re.compile(r'!\[([^\]]*)\]\(\s*(<[^>]*>|[^)\s]+)(?:\s+"[^"]*"|\s+\'[^\']*\'|\s+\([^)]+\))?\s*\)')
_MD_IMAGE_REF   = re.compile(r'!\[([^\]]*)\]\s*\[([^\]]*)\]')
_MD_LINK        = re.compile(r'\[([^\]]+)\]\(\s*(<[^>]*>|[^)\s]+)(?:\s+"[^"]*"|\s+\'[^\']*\'|\s+\([^)]+\))?\s*\)')
_MD_REF_DEF     = re.compile(r'^\s{0,3}\[([^\]]+)\]:\s*(<[^>]+>|[^ \t]+)(?:[ \t]+(?:"[^"]*"|\'[^\']*\'|\([^)]+\)))?\s*$',
                             re.MULTILINE)
_AUTOLINK       = re.compile(r'<(https?://[^>\s]+)>')
_BARE_URL       = re.compile(r'(?<!\()(?<!\])\b(?:https?|ftp|mailto):\/\/[^\s)>\]]+')
_DATA_URI       = re.compile(r'\bdata:[a-zA-Z]+/[a-zA-Z0-9+.\-]+;base64,[A-Za-z0-9+/=]+', re.IGNORECASE)

def _strip_brackets(u: str) -> str:
    u = u.strip()
    if u.startswith("<") and u.endswith(">"):
        return u[1:-1].strip()
    return u

def _is_long_or_data(u: str) -> bool:
    u = _strip_brackets(u)
    return u.lower().startswith("data:") or len(u) > MAX_URL_CHARS

def sanitize_export(text: str) -> str:
    # 1) Remove images (keep alt text only)
    text = _MD_IMAGE_INLINE.sub(lambda m: (m.group(1) or ""), text)
    text = _MD_IMAGE_REF.sub(lambda m: (m.group(1) or ""), text)

    # 2) Replace markdown links with just their labels if URL is long or data: URI
    def _md_link_sub(m):
        label, url = m.group(1), m.group(2)
        return label if _is_long_or_data(url) else m.group(0)
    text = _MD_LINK.sub(_md_link_sub, text)

    # 3) Drop reference-style definitions whose target is long or data:
    def _ref_def_sub(m):
        url = m.group(2)
        return "" if _is_long_or_data(url) else m.group(0)
    text = _MD_REF_DEF.sub(_ref_def_sub, text)

    # 4) Remove autolinks and bare URLs that are long or data:
    text = _AUTOLINK.sub(lambda m: "" if _is_long_or_data(m.group(1)) else m.group(0), text)
    text = _BARE_URL.sub(lambda m: "" if _is_long_or_data(m.group(0)) else m.group(0), text)

    # 5) Remove any remaining data: URIs (defense-in-depth)
    text = _DATA_URI.sub("", text)

    # Normalize multiple blank lines that deletions can create
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def build_drive_service() -> "googleapiclient.discovery.Resource":
    """Build and return the Google Drive API service with authentication."""
    creds = None
    token_path = "token.json"
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            #creds = flow.run_local_server(host="127.0.0.1", port=53682, open_browser=False, prompt="consent")
            # Use out-of-band flow for remote/Tailscale setups (no callback server needed)
            flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
            auth_url, _ = flow.authorization_url(prompt="consent")
            
            print("\n" + "="*70)
            print("AUTHENTICATION REQUIRED")
            print("="*70)
            print("\n1. Open this URL in your browser:\n")
            print(f"   {auth_url}\n")
            print("2. Authorize the application")
            print("3. Copy the authorization code from the browser\n")
            print("="*70)
            
            code = input("Enter the authorization code: ").strip()
            flow.fetch_token(code=code)
            creds = flow.credentials
            
        with open(token_path, "w", encoding="utf-8") as token:
            token.write(creds.to_json())
    
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_google_docs(service, created_after_iso=None):
    query = (
        "mimeType='application/vnd.google-apps.document' "
        "and trashed = false "
        "and 'me' in owners"
    )
    if created_after_iso:
        query += f" and createdTime >= '{created_after_iso}'"
    fields = "nextPageToken, files(id, name)"
    page_token = None
    while True:
        resp = service.files().list(
            q=query,
            fields=fields,
            pageSize=1000,
            pageToken=page_token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        for f in resp.get("files", []):
            yield {"id": f["id"], "name": f["name"]}
        page_token = resp.get("nextPageToken")
        if not page_token:
            break


def export_doc_bytes(service, file_id: str, mime_type: str, max_retries: int = 5) -> bytes:
    """
    Exports the given Google Doc to the requested mime_type and returns bytes.
    Retries on 429/5xx with exponential backoff.
    """
    attempt = 0
    while True:
        try:
            return service.files().export(fileId=file_id, mimeType=mime_type).execute()
        except HttpError as e:
            status = getattr(e, "status_code", None) or (e.resp.status if hasattr(e, "resp") else None)
            attempt += 1
            if attempt > max_retries or not status or status < 500:
                raise
            sleep_s = min(2 ** attempt, 30)
            time.sleep(sleep_s)


def safe_filename(title: str, used: set, extension: str, file_id: str) -> str:
    """
    Sanitizes the title into a filesystem-safe name, ensures uniqueness,
    and appends the extension. Uses a short id suffix on collision.
    """
    if not title:
        title = "Untitled"

    # Replace characters not allowed on common filesystems
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", title).strip()
    # Avoid trailing dots/spaces on Windows
    name = name.rstrip(" .")

    # Truncate to avoid exceeding common filename length limits
    if len(name) > MAX_BASENAME_LEN:
        name = name[:MAX_BASENAME_LEN].rstrip()

    candidate = f"{name}{extension}"
    # Ensure uniqueness
    if candidate in used:
        short_id = file_id[:8]
        candidate = f"{name} - {short_id}{extension}"
        i = 2
        while candidate in used:
            candidate = f"{name} - {short_id}-{i}{extension}"
            i += 1
    used.add(candidate)
    return candidate


def write_file(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def main():
    parser = argparse.ArgumentParser(description="Export all Google Docs to local files.")
    parser.add_argument("base_dir", help="Base directory to create the 'gdocs' folder in.")
    parser.add_argument("--years", type=int, default=5, help="Export only documents created in the last N years (default: 5).")
    parser.add_argument(
        "--exclude_formatting",
        action="store_true",
        help="Don't export as Markdown.",
    )
    args = parser.parse_args()

    out_dir = os.path.join(args.base_dir, "gdocs")
    os.makedirs(out_dir)

    export_mime = "text/markdown" if not args.exclude_formatting else "text/plain"
    extension = ".txt" #".md" if not args.exclude_formatting else ".txt"

    cutoff = datetime.now(timezone.utc) - timedelta(days=args.years*365)  # ~5 years
    cutoff_iso = cutoff.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    service = build_drive_service()

    used_filenames = set()
    exported = 0
    failed: List[Tuple[str, str]] = []

    docs = list(list_google_docs(service, created_after_iso=cutoff_iso))
    for doc in tqdm(docs, desc="Exporting Google Docs", unit="doc"):
        file_id = doc["id"]
        title = doc["name"]
        
        if title.startswith("Untitled document"):
            continue  # Skip untitled documents
        
        filename = safe_filename(title, used_filenames, extension, file_id)
        dest_path = os.path.join(out_dir, filename)

        try:
            try:
                content = export_doc_bytes(service, file_id, export_mime)
            except HttpError:
                # Fallback to plain text if markdown export isn't supported
                if not args.exclude_formatting:
                    content = export_doc_bytes(service, file_id, "text/plain")
                else:
                    raise
                
            text = content.decode("utf-8", errors="replace")
            text = sanitize_export(text)
            write_file(dest_path, text.encode("utf-8"))
            exported += 1
        except Exception as e:
            failed.append((title, str(e)))

    print(f"Exported {exported} document(s) to: {out_dir}")
    if failed:
        print("Failed exports:", file=sys.stderr)
        for title, err in failed:
            print(f"  - {title}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
