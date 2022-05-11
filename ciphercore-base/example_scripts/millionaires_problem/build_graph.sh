#!/bin/sh

set -xe
ciphercore_millionaires > ciphercore-base/example_scripts/millionaires_problem/plain_graph.json && ciphercore_compile ciphercore-base/example_scripts/millionaires_problem/plain_graph.json simple 0,1 0,1 > ciphercore-base/example_scripts/millionaires_problem/mpc_graph.json
