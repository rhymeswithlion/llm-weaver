#!/usr/bin/env bash

THIS_SCRIPT_DIR=$(dirname $0)

# Remove the .venv/bin directory from the PATH environment variable if it is there
if [[ $PATH =~ $HOME\/.venv\/bin ]]; then
    export PATH=$(echo $PATH | sed -e "s/:$THIS_SCRIPT_DIR\/.venv\/bin//g" -e "s/^$THIS_SCRIPT_DIR\/.venv\/bin://g")
    echo "Deactivated virtual environment by modifying PATH: $PATH"
else
    echo "Virtual environment already deactivated"
fi

echo "Python interpreter: $(which python)"
echo "Python version: $(python --version)"
echo "Git user: $(git config user.name)"
echo "Git email: $(git config user.email)"
