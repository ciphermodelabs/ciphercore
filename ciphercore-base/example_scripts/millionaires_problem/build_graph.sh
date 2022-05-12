#!/bin/sh

set -xe
ciphercore_millionaires > plain_graph.json && ciphercore_compile plain_graph.json simple 0,1 0,1 > mpc_graph.json
