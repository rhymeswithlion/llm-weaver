# CS194/294 Awesome Group Project

To see if your system is working

First, clean up any previous virtualenvs in .venv

```
$ make clean
=========================
 * Cleaning up
rm -rf .venv
```

Now create a new virtualenv and run the script `model_merging_isometric_fast.sh`

```
$ make .venv run-isometric-test

=========================
 * Creating conda environment
conda env create --prefix .venv/ -f environment.yml
Channels:
 - pytorch
 - conda-forge
 - defaults
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: done

Downloading and Extracting Packages:

Preparing transaction: done
Verifying transaction: done
Executing transaction: done
Installing pip dependencies: / 

```
