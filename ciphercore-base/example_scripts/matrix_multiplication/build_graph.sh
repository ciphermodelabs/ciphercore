#!/bin/sh

set -xe
ciphercore_matrix_multiplication -s i32 2 3 4 > ciphercore-base/example_scripts/matrix_multiplication/plain_graph.json && ciphercore_compile ciphercore-base/example_scripts/matrix_multiplication/plain_graph.json simple 1,2 0 > ciphercore-base/example_scripts/matrix_multiplication/mpc_graph.json