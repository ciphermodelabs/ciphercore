#!/bin/bash

set -xe

docker run -d --network host -it --name data_node0 -v "$WORK_DIR":/mnt/external --rm  ciphermodelabs/runtime_example:latest /usr/local/ciphercore/run.sh data_node 0.0.0.0 4201 4202 data/party0/input.txt certificates/ca.pem certificates/party0/data_cert.pem certificates/party0/data_key.pem
docker run -d --network host -it --name data_node1 -v "$WORK_DIR":/mnt/external --rm  ciphermodelabs/runtime_example:latest /usr/local/ciphercore/run.sh data_node 0.0.0.0 4211 4212 data/party1/input.txt certificates/ca.pem certificates/party1/data_cert.pem certificates/party1/data_key.pem
docker run -d --network host -it --name data_node2 -v "$WORK_DIR":/mnt/external --rm  ciphermodelabs/runtime_example:latest /usr/local/ciphercore/run.sh data_node 0.0.0.0 4221 4222 data/party2/input.txt certificates/ca.pem certificates/party2/data_cert.pem certificates/party2/data_key.pem
