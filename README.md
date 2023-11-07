# CS194/294 Awesome Group Project

This is our shared workspace!

The best way to use this:

- 1. Open the workspace file (`2023-fall-cs-194-294-merging-llms.code-workspace`) in Visual Studio Code
- 2. Run the following in the terminal:
  - `brew install miniconda` to install the conda command line tool
  - `make dev-setup` to set up the virtual environment in .venv
- 3. Install the needed Visual Studio Code extensions (Hint: choose "Extensions: Show Recommended Extensions" from the Command Palette, Cmd+Shift+P):
  - Python
  - Jupyter
- 4. Open `notebooks/example-notebook-distilbert-classifier-fine-tuned.ipynb` and run it. If it asks for a "kernel" use the recommended one in `.venv/bin/python3`. It should work!
- 5. Run the scripts in `scripts/` from the terminal. They should work!

Notes for running Python commands in the terminal:

- If you want to activate the virtualenv in your terminal, run `source source_to_activate.sh`. This should happen automatically with vscode, if you opened the workspace in step 1.
