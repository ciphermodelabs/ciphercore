#!/bin/bash

set -ex


python -m venv /tmp/test-env
source /tmp/test-env/bin/activate

pip install setuptools_rust
pip uninstall ciphercore

python setup.py install
python a.py

deactivate
