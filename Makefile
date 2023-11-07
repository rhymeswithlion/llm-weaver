.PHONY=dev-setup pre-commit-run clean print-conda-environment

# Set bash as the shell
SHELL := /bin/bash

all: dev-setup

dev-setup:
	make .venv
	# Install pre-commit hooks
	.venv/bin/pre-commit install

	cd research && make all

pre-commit-run:
	.venv/bin/pre-commit run --all-files

.venv:
	@echo "Creating conda environment"
	@export CONDA_PKGS_DIRS=$(shell mkdir -p .venv/packages && realpath .venv/packages) \
	&& export CONDA_ENV_DIR=$(shell mkdir -p .venv && realpath .venv) \
	&& conda create --prefix $$CONDA_ENV_DIR python=3.8.18 --yes \
	&& conda env update --prefix $$CONDA_ENV_DIR --file environment.yml --prune

clean:
	rm -rf .venv

print-conda-environment:
	conda env export --prefix .venv/ | grep -v "^prefix: "
