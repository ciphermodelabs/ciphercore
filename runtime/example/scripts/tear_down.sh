#!/bin/bash

set -xe

docker kill data_node0 || true
docker kill data_node1 || true
docker kill data_node2 || true

docker kill compute_node0 || true
docker kill compute_node1 || true
docker kill compute_node2 || true
