# Gmail Training Data Export

Export your Gmail conversations to create training data for fine-tuning an LLM to write emails in your style.

## Overview

This system extracts email conversations where:
- **Input (User)**: Emails you received from others (boss, colleagues, clients)
- **Output (Assistant)**: Your responses to those emails

The trained model can then help you draft email responses in your personal writing style.

## Setup

### 1. Google Cloud Project Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Gmail API**:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Gmail API"
   - Click "Enable"

### 2. OAuth Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. Configure the OAuth consent screen if prompted
4. Choose "Desktop app" as the application type
5. Download the credentials JSON file
6. Save it as `credentials.json` in the project root

### 3. Install Dependencies

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client tqdm python-dotenv
```

Or if you have a requirements.txt:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Export Emails

Export your Gmail threads to JSON files:

```bash
python3 gmail/export_gmail.py data_alex --years 5
```

**Arguments:**
- `data_alex`: Base directory where `emails/` folder will be created
- `--years`: Number of years of email history to export (default: 5)

**What it does:**
- Authenticates with Gmail (opens browser for first-time auth)
- Fetches all email threads from the specified time period
- Exports each thread as a JSON file in `data_alex/emails/`
- Skips spam, trash, and chat messages
- Only includes threads with back-and-forth conversation

**First run:** The script will open a browser window asking you to authorize access to your Gmail. After authorization, a `token.json` file is created for future runs.

### Step 2: Convert to Training Format

Process the exported emails into JSONL training data:

```bash
python3 gmail/to_final_file.py data_alex --output final_gmail.jsonl
```

**Arguments:**
- `data_alex`: Directory containing the `emails/` folder
- `--output`: Output filename (default: `final_gmail.jsonl`)

**What it does:**
- Reads all thread JSON files from `data_alex/emails/`
- Extracts conversation pairs where you responded to someone
- Creates training examples with proper context
- Outputs JSONL file: `data_alex/final_gmail.jsonl`

### Output Format

Each line in the JSONL file contains:

```json
{
  "input": "From: boss@company.com\nSubject: Project Update Needed\n\nHi, can you send me the latest status on the project?\n\nThanks",
  "output": "Hi Boss,\n\nAbsolutely! Here's the current status:\n- Milestone 1: Complete\n- Milestone 2: 80% done, on track\n- Next steps: Review meeting on Friday\n\nLet me know if you need more details.\n\nBest,\nYour Name"
}
```

## Data Flow

```
Gmail Account
    ↓
export_gmail.py → data_alex/emails/thread_*.json
    ↓
to_final_file.py → data_alex/final_gmail.jsonl
    ↓
Fine-tuning Script
```

## Features

### Email Cleaning
- Removes quoted replies ("> quoted text")
- Strips email signatures and footers
- Removes disclaimers and auto-generated content
- Keeps only the actual message content

### Context Preservation
- Includes full email thread context in the input
- Shows previous messages in the conversation
- Preserves subject line and sender information
- Helps model understand conversation flow

### Privacy & Security
- All data stays local on your machine
- OAuth tokens stored in `token.json`
- Read-only access to Gmail (can't modify emails)
- No data sent to third parties

## Training the Model

After generating `final_gmail.jsonl`, you can use it with your existing fine-tuning pipeline:

```bash
# Example: Using with your existing finetune.py
python3 finetune.py --data data_alex/final_gmail.jsonl --model <base_model>
```

The model will learn:
- Your writing style and tone
- How you structure emails
- Common phrases and sign-offs you use
- How you respond to different types of requests

## Tips

### For Better Results

1. **More data is better**: Export 3-5 years of email history
2. **Filter by sender**: Modify the export script to focus on important contacts
3. **Combine with other data**: Merge with your Google Docs or iMessage data
4. **Review samples**: Check a few training pairs to ensure quality

### Customization

Edit `export_gmail.py` to:
- Filter by specific senders or labels
- Change the date range
- Modify what metadata to include
- Adjust email cleaning rules

Edit `to_final_file.py` to:
- Change the formatting of input/output
- Add system prompts
- Filter by email length or complexity
- Customize the training pair extraction logic

## Troubleshooting

**"Credentials not found"**
- Make sure `credentials.json` is in the project root
- Re-download from Google Cloud Console if needed

**"Invalid grant" error**
- Delete `token.json` and re-authenticate
- Make sure OAuth consent screen is configured

**No threads found**
- Check the date range (--years parameter)
- Verify you have emails in the specified period
- Check that emails aren't all in spam/trash

**Low training pairs**
- You may have mostly one-way emails
- Try increasing --years to get more data
- Check that emails have actual conversation exchanges

## License

This code follows the same license as the parent project.
