#!/bin/bash

set -xe

echo "Creating workdir..."
export WORK_DIR="$1"
mkdir -p "$WORK_DIR"
mkdir -p "$WORK_DIR"/data

echo "Pulling the docker..."
docker login -u runtimecm
docker pull ciphermodelabs/runtime_example:latest

echo "Generating tls certificates..."
docker run --rm -u $(id -u):$(id -g) -v "$WORK_DIR":/mnt/external ciphermodelabs/runtime_example:latest /usr/local/ciphercore/run.sh generate_certs certificates/ localhost

echo "Copying the median example data..."
cp -rf runtime/example/data/two_millionaires/* "$WORK_DIR"/data/

echo "Starting data nodes..."
sh runtime/example/scripts/run_data_nodes.sh || sh runtime/example/scripts/tear_down.sh

echo "Starting compute nodes..."
sh runtime/example/scripts/run_compute_nodes.sh || sh runtime/example/scripts/tear_down.sh

echo "Performing computation..."
sh runtime/example/scripts/run_orchestrator.sh || sh runtime/example/scripts/tear_down.sh

echo "Releasing all processes..."
sh runtime/example/scripts/tear_down.sh

echo "Done."
