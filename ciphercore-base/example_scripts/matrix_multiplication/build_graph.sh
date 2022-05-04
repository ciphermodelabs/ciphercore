#!/bin/sh

set -xe
./target/release/matrix_multiplication -s i32 2 3 4 | ./target/release/compile simple 1,2 0 > ciphercore-base/example_scripts/matrix_multiplication/mpc_graph.json