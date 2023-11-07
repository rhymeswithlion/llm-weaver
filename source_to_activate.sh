#!/usr/bin/env bash

THIS_SCRIPT_DIR=$(dirname $0)

# Prepand the .venv/bin directory to the PATH environment variable if it is not already there
if [[ ! $PATH =~ $THIS_SCRIPT_DIR\/.venv\/bin ]]; then
    export PATH="$THIS_SCRIPT_DIR/.venv/bin:$PATH"
    echo "Activated virtual environment by modifying PATH: $PATH"
else
    echo "Virtual environment already activated"
fi

echo "Python interpreter: $(realpath $(which python))"
echo "Python version: $(python --version)"
echo "Git user: $(git config user.name)"
echo "Git email: $(git config user.email)"
