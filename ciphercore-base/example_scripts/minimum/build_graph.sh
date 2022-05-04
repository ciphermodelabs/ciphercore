#!/bin/sh

set -xe
./target/release/minimum -s i32 4 | ./target/release/compile depth-optimized-default 0,1 1 > ciphercore-base/example_scripts/minimum/mpc_graph.json