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
