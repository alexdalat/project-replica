
import pandas as pd
import json
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# getenv
import os
from dotenv import load_dotenv
load_dotenv()


SAME_CONVO_THRESHOLD_SECONDS = 3600
SAME_USER_THRESHOLD_SECONDS = 600
HISTORY_MAX_TOKENS = 3000
CONVO_MIN_TOKENS = 100


# create the tokenizer to measure the length of the text
base_model_id = os.getenv("base_model_id")
if not base_model_id:
    base_model_id = "alpindale/Mistral-7B-v0.2-hf"
    print(f"base_model_id not set, using default: {base_model_id}")
encoder = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=False, trust_remote_code=True, use_fast=True, force_download=False)  #add_special_tokens=False,

# combine messages from the same sender within 5 mins into a single new-line separated message
def collapse_messages(df, delta_threshold=SAME_USER_THRESHOLD_SECONDS):
    if len(df) == 0:
        return df
    
    new_data = []
    
    df_temp = df.copy()
    current_row = df_temp.iloc[0]
    current_role = current_row["chat_message"][0]

    for _, row in tqdm(df_temp[1:].iterrows(), total=len(df_temp)-1):
        row_role = row["chat_message"][0]
        row_message = row["chat_message"][1]

        if row_role == current_role and row["time_delta"] < delta_threshold:
            current_row["chat_message"] = (current_row["chat_message"][0], current_row["chat_message"][1] + "\n" + row_message)
        else:
            new_data.append(current_row.to_dict())
            current_row = row
            current_role = row_role
    
    # add last row
    new_data.append(current_row.to_dict())

    return pd.DataFrame(new_data)



import re
from datetime import datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json

LINE_RE = re.compile(
    r'^(\d{2}/\d{2}/\d{4}), (\d{2}:\d{2}) - ([^:]+): (.*)$'
)
# Example it matches:
# 07/06/2024, 13:28 - Watson: Adult price 15.50 per person omg

def parse_thread_txt(filepath: Path) -> pd.DataFrame:
    """Parse our converted iMessage thread into a DataFrame with
    columns: date (datetime), username (str), message (str).
    Supports multi-line messages.
    """
    rows = []
    pending = None  # (dt, user, msg_so_far)

    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = LINE_RE.match(line)
            if m:
                # flush previous pending
                if pending is not None:
                    dt, user, msg = pending
                    rows.append({"date": dt, "username": user, "message": msg.strip()})
                # start new one
                d_str, t_str, user, msg = m.groups()
                dt = datetime.strptime(f"{d_str} {t_str}", "%d/%m/%Y %H:%M")
                pending = (dt, user.strip(), msg)
            else:
                # continuation of previous message (multi-line)
                if pending is not None:
                    dt, user, msg = pending
                    pending = (dt, user, msg + "\n" + line)
                # otherwise ignore stray junk

    # flush last
    if pending is not None:
        dt, user, msg = pending
        rows.append({"date": dt, "username": user, "message": msg.strip()})

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("date").reset_index(drop=True)
    return df

def preprocess_convo(input_path, output_path, role="user", chat_owner="Watson"):
    # Build the df from our simple TXT format
    df = parse_thread_txt(Path(input_path))
    if df.empty:
        # nothing to write
        with open(output_path, "w") as f:
            pass
        return

    # Calculate time passed since previous message
    df["date_previous"] = df["date"].shift(1)
    df["time_delta"] = (df["date"] - df["date_previous"]).dt.total_seconds()

    # Map owner to 'assistant', others to role
    df["chat_message"] = df.apply(
        lambda x: ("assistant" if x["username"] == chat_owner else role, x["message"]),
        axis=1
    )

    # collapse consecutive messages from the same speaker if you had that helper
    # If you don't have collapse_messages(), comment the next line or implement your own.
    df = collapse_messages(df)

    query = []
    conversation = []
    token_len = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        row_role = row["chat_message"][0]
        row_message = row["chat_message"][1]

        chat_message_formatted = f"<start_header_id>{row_role}<end_header_id>{row_message}"
        chat_message_formatted_len = len(encoder.encode(chat_message_formatted))

        if row["time_delta"] < SAME_CONVO_THRESHOLD_SECONDS and \
           token_len + chat_message_formatted_len < HISTORY_MAX_TOKENS:
            conversation.append(chat_message_formatted)
            token_len += chat_message_formatted_len
        else:
            if conversation:
                query.append(conversation)
            conversation = [chat_message_formatted]
            token_len = chat_message_formatted_len

    if conversation:
        query.append(conversation)

    df_model = pd.DataFrame({"query": query})
    df_model["query_str"] = df_model["query"].apply(lambda x: "<|eot_id|>".join(x))
    df_model["query_len"] = df_model["query_str"].apply(lambda x: len(encoder.encode(x)))

    df_model_filtered = df_model[df_model["query_len"] > CONVO_MIN_TOKENS]

    with open(output_path, "w") as f:
        for _, row in df_model_filtered.iterrows():
            f.write(json.dumps({"input": row["query_str"]}) + "\n")
