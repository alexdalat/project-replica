# Project Replica
This project is designed to fine-tune a language model to mimic your own writing style, using your iMessage history as training data.

- [Project Replica](#project-replica)
	- [Installation](#installation)
		- [Requirements](#requirements)
		- [Inputs](#inputs)
		- [Running](#running)
		- [Outputs](#outputs)
	- [References](#references)

## Installation

### Requirements
 - Python 3.12+ (previous versions may work, but not tested)
 - built llama.cpp at project root (for model deployment to Ollama)
 
### Inputs
(replace text in braces with you own values)
 - data_{NAME} - folder containing `chat.db` (from Messages) and `Sources` (from AddressBook)

### Running
```bash
git clone https://github.com/alexdalat/project-replica
cd project-replica
pip install -r requirements.txt
cp ~/Library/Messages/chat.db data_$USER/chat.db  # or copy these over manually
cp ~/Library/Application\ Support/AddressBook/Sources/ data_$USER/
chmod +x run_all.sh
./export_imessages.sh data_$USER
python3 finetune.py data_$USER/final.jsonl models/$USER/imsg_$(date +'%Y-%m-%d_%H-%M') --dataset_limit=500 --epochs=1 --batch_size=1
```

### Outputs
 - `models/` - folder with newly created fine-tuned models
 - `logs/` - tensorboard outputs that can be seen with `tensorboard --logdir=logs`


## References
 - https://medium.com/@watsonchua/finetuning-my-clone-training-an-llm-to-talk-like-me-2ee7b5ba2f88 and its GitHub
 - https://medium.com/@yuxiaojian/fine-tuning-llama3-1-and-deploy-to-ollama-f500a6579090