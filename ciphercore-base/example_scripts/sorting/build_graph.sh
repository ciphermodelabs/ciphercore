#!/bin/sh

set -xe
./target/release/batcher_sorting -s i32 4 | ./target/release/compile depth-optimized-default 0 1 > ciphercore-base/example_scripts/sorting/mpc_graph.json
