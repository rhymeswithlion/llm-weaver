.PHONY=dev-setup pre-commit-run clean print-conda-environment

# Set bash as the shell
SHELL := /bin/bash

all: dev-setup

dev-setup:
	make .venv

pre-commit-run:
	.venv/bin/pre-commit run --all-files

.venv:
	@echo "Creating conda environment"
	conda env create --prefix .venv/ -f environment.yml

clean:
	rm -rf .venv

print-conda-environment:
	conda env export --prefix .venv/ | grep -v "^prefix: "
