# CS194/294 Awesome Group Project

To see if your system is working

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
If you want to delete the virtualenv run:

```
$ make clean
=========================
 * Cleaning up
rm -rf .venv
```