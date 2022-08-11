#!/bin/bash

set -xe

docker run --network host -it --name do -v "$WORK_DIR":/mnt/external --rm  ciphermodelabs/runtime_example:latest /usr/local/ciphercore/run.sh orchestrator data/graph.json 127.0.0.1 4203 127.0.0.1 4213 127.0.0.1 4223 certificates/ca.pem certificates/orchestrator_cert.pem certificates/orchestrator_key.pem

echo "Party 0 got result:"
docker logs data_node0 --tail 20
echo "Party 1 got result:"
docker logs data_node1 --tail 20
echo "Party 2 got result:"
docker logs data_node2 --tail 20
