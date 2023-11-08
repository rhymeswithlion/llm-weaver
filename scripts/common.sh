#!/usr/bin/env bash

set -e

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PATH=$SCRIPTS_DIR/../.venv/bin:$PATH
export PYTHONPATH=$SCRIPTS_DIR/../model_merging:$PYTHONPATH

# If Python found at $SCRIPTS_DIR/../.venv/bin/python, then use it.
echo "Using Python from $(which python)"

python -c "import model_merging; print('model_merging package found: ' +  str(model_merging.__file__))"
