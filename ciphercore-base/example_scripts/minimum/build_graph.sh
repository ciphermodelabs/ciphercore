#!/bin/sh

set -xe
ciphercore_minimum -s i32 4 > ciphercore-base/example_scripts/minimum/plain_graph.json && ciphercore_compile ciphercore-base/example_scripts/minimum/plain_graph.json depth-optimized-default 0 1 > ciphercore-base/example_scripts/minimum/mpc_graph.json
