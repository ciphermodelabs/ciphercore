#!/bin/sh

set -xe
export LD_LIBRARY_PATH=$(./onnx.sh)
./target/release/plaintext_evaluator_main ciphercore-base/example_scripts/millionaires_problem/mpc_graph.json ciphercore-base/example_scripts/millionaires_problem/inputs.json
