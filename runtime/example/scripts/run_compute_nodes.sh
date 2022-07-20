#!/bin/bash

set -xe

docker run -d --network host -it --name compute_node0 -v "$WORK_DIR":/mnt/external --rm  ciphermodelabs/runtime_example:latest /usr/local/ciphercore/run.sh compute_node 0.0.0.0 4203 127.0.0.1 4201 127.0.0.1 4202 certificates/ca.pem certificates/party0/cert.pem certificates/party0/key.pem localhost
docker run -d --network host -it --name compute_node1 -v "$WORK_DIR":/mnt/external --rm  ciphermodelabs/runtime_example:latest /usr/local/ciphercore/run.sh compute_node 0.0.0.0 4213 127.0.0.1 4211 127.0.0.1 4212 certificates/ca.pem certificates/party1/cert.pem certificates/party1/key.pem localhost
docker run -d --network host -it --name compute_node2 -v "$WORK_DIR":/mnt/external --rm  ciphermodelabs/runtime_example:latest /usr/local/ciphercore/run.sh compute_node 0.0.0.0 4223 127.0.0.1 4221 127.0.0.1 4222 certificates/ca.pem certificates/party2/cert.pem certificates/party2/key.pem localhost
