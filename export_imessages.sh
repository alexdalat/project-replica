
# check that $1 is set
if [ -z "$1" ]; then
	echo "Usage: $0 <data_directory>"
	exit 1
fi

# check that $1 exists
if [ ! -d "$1" ]; then
	echo "Directory $1 does not exist."
	exit 1
fi
# check that $1/Sources/**/*.abcddb exists
if [ -z "$(find $1/Sources -name '*.abcddb' -print -quit)" ]; then
	echo "No .abcddb files found in $1/Sources."
	exit 1
fi
# check that $1/chat.db exists
if [ ! -f "$1/chat.db" ]; then
	echo "File $1/chat.db does not exist."
	exit 1
fi

python3 messages/get_addressbook.py $1  # not really needed for production
python3 messages/export_imessages.py $1
python3 messages/to_final_file.py $1

echo -e "When you're ready, run \n\n\
python3 finetune.py $1/final.jsonl models/\$USER --dataset_limit=500 --epochs=1 --batch_size=1\n\n\
to start the fine-tuning process."
echo -e "\nAfter fine-tuning, you can export to Ollama with:\n"
echo -e "./ollama_export.sh models/\$USER/imsg_YYYY-MM-DD_HH-MM\n"