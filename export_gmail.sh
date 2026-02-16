#!/bin/bash
# Export Gmail conversations and convert to training data

# check that $1 is set
if [ -z "$1" ]; then
	echo "Usage: $0 <data_directory> [years]"
	echo "Example: $0 data_alex 5"
	exit 1
fi

# check that $1 exists, if not create it
if [ ! -d "$1" ]; then
	echo "Directory $1 does not exist. Creating it..."
	mkdir -p "$1"
fi

# Optional: years parameter (default to 5)
YEARS=${2:-5}

echo "Exporting Gmail conversations from the last $YEARS years..."
python3 gmail/export_gmail.py "$1" --years "$YEARS"

if [ $? -ne 0 ]; then
	echo "Error during Gmail export. Exiting."
	exit 1
fi

echo "Converting emails to training data..."
python3 gmail/to_final_file.py "$1"

if [ $? -ne 0 ]; then
	echo "Error during conversion. Exiting."
	exit 1
fi

echo -e "\n✓ Gmail export complete!"
echo -e "\nWhen you're ready, run:\n"
echo -e "python3 finetune.py $1/final_gmail.jsonl models/\$USER --dataset_limit=500 --epochs=1 --batch_size=1\n"
echo -e "to start the fine-tuning process."
echo -e "\nAfter fine-tuning, you can export to Ollama with:\n"
echo -e "./ollama_export.sh models/\$USER/gmail_YYYY-MM-DD_HH-MM\n"
