#!/bin/bash
set -ex

curl https://sh.rustup.rs -sSf | sh -s -- -y
export PATH="$HOME/.cargo/bin:$PATH"

for PYBIN in /opt/python/{cp37-cp37m,cp38-cp38,cp39-cp39,cp310-cp310}/bin; do
    export PYTHON_SYS_EXECUTABLE="$PYBIN/python"

    "${PYBIN}/pip" install -U setuptools-rust==0.11.3
    "${PYBIN}/python" setup.py bdist_wheel
    rm -rf build/*
done

for whl in dist/*.whl; do
    auditwheel repair "$whl" -w dist/
done
