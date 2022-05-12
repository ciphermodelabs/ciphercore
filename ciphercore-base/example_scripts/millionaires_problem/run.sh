#!/bin/sh

set -xe
export LD_LIBRARY_PATH=$(./onnx.sh)
ciphercore_evaluate mpc_graph.json inputs.json
