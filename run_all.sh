
# check that $1 is set
if [ -z "$1" ]; then
	echo "Usage: $0 <data_directory>"
	exit 1
fi

# check that data_$1 exists
if [ ! -d "data_$1" ]; then
	echo "Directory data_$1 does not exist."
	exit 1
fi
# check that data_$1/Sources/**/*.abcddb exists
if [ -z "$(find data_$1/Sources -name '*.abcddb' -print -quit)" ]; then
	echo "No .abcddb files found in data_$1/Sources."
	exit 1
fi
# check that data_$1/chat.db exists
if [ ! -f "data_$1/chat.db" ]; then
	echo "File data_$1/chat.db does not exist."
	exit 1
fi

python3 messages/get_addressbook.py data_$1  # not really needed for production
python3 messages/export_imessages.py data_$1
python3 messages/to_final_file.py data_$1

echo -e "When you're ready, run \n\n\
python3 finetune.py data_$1/final.jsonl models/$1/imsg_\$(date +'%Y-%m-%d_%H-%M')/\n\n\
to start the fine-tuning process."