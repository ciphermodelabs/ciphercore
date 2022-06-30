#!/bin/sh

set -xe
ciphercore_set_intersection -s i32 4 > plain_graph.json && ciphercore_compile plain_graph.json depth-optimized-default 0,1 1 > mpc_graph.json
