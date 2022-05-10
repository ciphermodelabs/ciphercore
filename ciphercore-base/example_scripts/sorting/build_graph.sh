#!/bin/sh

set -xe
ciphercore_sorting -s i32 4 > ciphercore-base/example_scripts/sorting/plain_graph.json && ciphercore_compile ciphercore-base/example_scripts/sorting/plain_graph.json depth-optimized-default 0 1 > ciphercore-base/example_scripts/sorting/mpc_graph.json
