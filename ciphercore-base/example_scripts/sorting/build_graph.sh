#!/bin/sh

set -xe
ciphercore_sorting -s u32 4 > plain_graph.json && ciphercore_compile plain_graph.json depth-optimized-default 0 1 > mpc_graph.json
