#!/bin/sh

set -xe
ciphercore_matrix_multiplication -s i32 2 3 4 > plain_graph.json && ciphercore_compile plain_graph.json simple 1,2 0 > mpc_graph.json
