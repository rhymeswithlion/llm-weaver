.PHONY=.venv clean run-isometric-test

# Set bash as the shell
SHELL := /bin/bash

.venv:
	@echo "========================="
	@echo " * Creating conda environment"
	conda env create --prefix .venv/ -f environment.yml
	@echo "To activate with conda: conda activate $$PWD/.venv"
	
clean:
	@echo "========================="
	@echo " * Cleaning up"
	rm -rf .venv

run-isometric-test:
	@echo "========================="
	@echo " * Running isometric test"
	./scripts/model_merging_isometric_fast.sh

data/gpt2-finetuned-mnli-fixed:
	rm -rf data/gpt2-finetuned-mnli-fixed
	mkdir -p data/
	git lfs install
	cd data/ && git clone https://huggingface.co/PavanNeerudu/gpt2-finetuned-mnli gpt2-finetuned-mnli-fixed

	# Copy tokenizer.json to the data folder
	cp ./tokenizer.json data/gpt2-finetuned-mnli-fixed/tokenizer.json
	# Remove any lines with "pad_token" in tokenizer_config.json
	sed -i '' '/pad_token/d' data/gpt2-finetuned-mnli-fixed/tokenizer_config.json

data/gpt2-finetuned-rte-fixed:
	rm -rf data/gpt2-finetuned-rte-fixed
	mkdir -p data/
	git lfs install
	cd data/ && git clone https://huggingface.co/PavanNeerudu/gpt2-finetuned-rte gpt2-finetuned-rte-fixed

	cp ./tokenizer.json data/gpt2-finetuned-rte-fixed/tokenizer.json
	# Remove any lines with "pad_token" in tokenizer_config.json
	sed -i '' '/pad_token/d' data/gpt2-finetuned-rte-fixed/tokenizer_config.json
