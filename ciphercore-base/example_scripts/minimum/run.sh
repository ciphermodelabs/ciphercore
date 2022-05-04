#!/bin/sh

set -xe
export LD_LIBRARY_PATH=$(./onnx.sh)
./target/release/plaintext_evaluator_main ciphercore-base/example_scripts/minimum/mpc_graph.json ciphercore-base/example_scripts/minimum/inputs.json --reveal-output