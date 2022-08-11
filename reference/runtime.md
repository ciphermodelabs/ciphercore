# Runtime

For executing a compiled computation graph, one needs *runtime* that allows parties holding the inputs to perform the actual computation. Due to the nature of SMPC, one needs to run multiple instances of the runtime, which interactively communicate with each other over the network during the execution. The number of instances depends on the SMPC protocol which was used during the compilation stage; currently we support the [ABY3 protocol](https://eprint.iacr.org/2018/403.pdf), which requires three non-colluding parties.

**_NOTE_:** We don't provide the source code of the runtime in the open-source repository. If you want to execute graphs in runtime to measure real-life performance, we provide the binary/docker on request.

The runtime consists of three different entities: *compute nodes* (the runtime itself, which performs the computation), *data nodes* (services which provide the encrypted or plaintext data to the runtime, and consume the results of computation), and *orchestrator node* (controls the high-level structure of the execution: which computation graphs should be executed in which order and on which data).

![b](https://github.com/ciphermodelabs/ciphercore/blob/main/reference/images/runtime.svg)

## Pre-packaged runtime example
We provide a Docker container with a pre-packaged runtime which can be used to do the following: execute a given computation graph once on a given input. This can be useful for trying it out or benchmarking; however, real-life applications often require more complex logic.

TL;DR: there is a single [bash script](https://github.com/ciphermodelabs/ciphercore/blob/main/runtime/example/scripts/do_all.sh) which can run everything end-to-end.
```bash
sh runtime/example/scripts/do_all.sh /tmp/ciphercore
```
But it's possible to perform the steps manually as outlined below.

Note that docker commands might require `sudo` if the current user is not in the appropriate group.

For the networking we use 9 ports. For each party it is: "provide data" port (`42{#party}1`), "get result" port (`42{#party}2`), "computation" port (`42{#party}3`).
Feel free to change the scripts in order to use any other ports. (Please note that compute nodes should get the correct ports corresponding for the data and result.)
<details>
  <summary>All ports which are used</summary>
  ```
  4201, 4202, 4203, 4211, 4212, 4213, 4221, 4222, 4223
  ```
</details>

It is assumed that all scripts are running from the root of this repository (ciphercore/).
0. Set up your WORK_DIR:
```bash
export WORK_DIR="/tmp/ciphercore" # Or any other dir
mkdir -p "$WORK_DIR"
mkdir -p "$WORK_DIR"/data
```
1. Pull docker from our private repo ([contact us](https://www.ciphermode.tech/contact-us) to get the token).
    ```bash
    docker login -u runtimecm
    docker pull ciphermodelabs/runtime_example:latest
    ```
2. Generate [certificates](#certificates).
    ```bash
    docker run --rm -u $(id -u):$(id -g) -v "$WORK_DIR":/mnt/external ciphermodelabs/runtime_example:latest /usr/local/ciphercore/run.sh generate_certs certificates/ localhost
    ```
    `localhost` — this is the TLS domain, in real-world deployments, the runtime wouldn't work if domains don't match, but for the purposes of the example, we always override it to localhost.
3. Generate computation graph and data.
    Computation could be performed via any graphs produced by the CipherCore compiler (see more about [compiling graphs](https://github.com/ciphermodelabs/ciphercore/blob/main/reference/main.md#working-with-the-graph-using-cli-tools)).
    We provide two examples:
    <details open><summary>the median of the sequence on the secret-shared input  (each party doesn't have access to the data);</summary>

    ```bash
    cp -rf runtime/example/data/median/* "$WORK_DIR"/data/
    ```
    </details>

    <details><summary>two millionaires problem with revealed input.</summary>

    ```bash
    cp -rf runtime/example/data/two_millionaires/* "$WORK_DIR"/data/
    ```
    </details>
    Note that there are two conceptually different cases: secret-shared inputs, and inputs provided by one of the parties. However, in both cases we ask all parties to provide the input — if they don't have it, they should provide any value of the correct type (e.g. zeros or a random value).
4. Start data nodes using the following script.
    ```bash
    sh runtime/example/scripts/run_data_nodes.sh
    ```
5. Start compute nodes using the following script.
    ```bash
    sh runtime/example/scripts/run_compute_nodes.sh
    ```
6. Run secure computation.
    ```bash
    sh runtime/example/scripts/run_orchestrator.sh
    ```
    After the end of the computation, the result for each party will be printed. Parties, which are not supposed to see the result, see random ``garbage''.

    Now you can also try to use different data/graphs without restarting everything. Just update the corresponding files in folder `$WORK_DIR/data` and call `run_orchestrator.sh` once more.

7. Once done, don't forget to stop all processes to release ports.
    ```bash
    sh runtime/example/scripts/tear_down.sh
    ```

## Implementation of application-specific data and orchestrator nodes

For more complex scenarios, one has to do some additional implementation work. We provide a universal binary for compute nodes, while the implementation of the data nodes and orchestrator(s) has to be application-specific. In most cases, the implementation is fairly straightforward. The interaction of data & orchestrator nodes with compute nodes is done via the following [gRPC](https://en.wikipedia.org/wiki/GRPC) interfaces (gRPC is available in many languages, not just Rust).

Specifications for the gRPC protocols can be found in the following files:

* Data nodes (providing inputs and consuming outputs): `runtime/proto/data.proto` and `runtime/proto/results.proto`
* Orchestrator: `runtime/proto/party.proto`.

The overall process is structured as follows. Each computation is represented as a *session*, which has an associated computation graph, and party-specific data keys. To perform the computation:

* Orchestrator first has to call the `RegisterGraph` RPC method of a compute node, to upload the computation graph to each compute node (once per graph, the same graph can be re-used in different sessions);
* Then, it has to call `CreateSession`, providing the id of the registered graph, as well as keys for data nodes (party-specific), and addresses of other compute nodes;
* Compute nodes retrieve their inputs from the data nodes, connect to each other, interactively run the computation protocol, and send the results back to data nodes;
* Once the computation is finished, the session should be removed with the `FinishSession` call to free up the memory. Multiple sessions can run in parallel.

Data nodes and orchestrator can be separate, or combined in any way. However, compute nodes must be separate, moreover,

**NO TWO COMPUTE NODES CAN BE CONTROLLED BY THE SAME ENTITY,**

otherwise this entity can decrypt the data.

## Certificates
An important part of runtime setup are TLS certificates. They specify which pairs of nodes can talk to each other, and make the communication secure via TLS encryption.

> **_NOTE:_**  The communication between compute nodes is already encrypted - most of the values they send to each other are indistinguishable from random noise by construction. However, this is not secure enough - if a certain party (e.g. router) gets access to all communication, they can reconstruct plaintext data. So an additional layer of encryption is required. 

Currently, the certificates are structured as follows. We generate our own root (CA) certificate. Then, we use it to issue certificates for all nodes. All nodes have the public key for CA, so they can validate the certificates of the nodes they're connecting to.

You can generate the certificates with the pre-packaged docker image:
```bash
docker run --rm -u $(id -u):$(id -g) -v "$WORK_DIR":/mnt/external ciphermodelabs/runtime_example:latest /usr/local/ciphercore/run.sh generate_certs certificates/ localhost
```

In real-life use-cases, sometimes an additional level of certificate hierarchy might be required. Imagine e.g. a cross-organization use-case: we might want to issue per-organization certificates, which organizations then use to issue certificates for their nodes. Note that there is an assumption that the owner of CA does not collude with any of the parties.

## Example of a custom runtime

For this example, we implement all three data nodes and orchestrator in a single Python script (not suitable for real-life deployments). We'll use the graph and inputs from the Rust example: the graph describes the computation of a median element in an array, and the input is this array secret-shared between the parties.

### Python-based data nodes and orchestrator

For the Python wrapper, one needs to generate the gRPC Python files from the proto files as follows (don't forget to install the `grpcio-tools` package):

```bash
python -m grpc_tools.protoc \
    -Iruntime/proto \
    --python_out=runtime/example/python \
    --grpc_python_out=runtime/example/python \
    runtime/proto/*.proto
```

This generates Python bindings for the proto files (a bunch of barely-readable Python files with `pb2` in their names).

You can find the ready-to-use example at `runtime/example/python/combined_example.py`. Below we go over it step-by-step and explain what is happening. 

First, let's implement the data nodes in Python. For each data node, we'll need to export two RPC services: for providing input data, and for consuming results. Both are straightforward:

```python
class KVDataServer(data_pb2_grpc.DataManagerServiceServicer):
    def __init__(self, kv_dict):
        super().__init__()
        self.kv_dict = kv_dict

    def GetValue(self, request, context):
        k = request.key
        vals = self.kv_dict.get(k)
        resp = data_pb2.GetValueResponse()
        if vals is None:
            resp.status = 1
            resp.error = 'Not found'
        else:
            resp.status = 0
            resp.typed_value.extend(vals)
        return resp


def parse_value(value_proto):
    if value_proto.HasField('bytes'):
        return cc.Value.from_bytes(value_proto.bytes.data)
    vec = []
    for sub_val in value_proto.vector.value:
        vec.append(parse_value(sub_val))
    return cc.Value.from_vector(vec)


def parse_typed_value(tv_proto):
    return cc.TypedValue(cc.Type.from_json_string(tv_proto.type_json), parse_value(tv_proto.value))


class TrivialResultServer(results_pb2_grpc.ResultProviderServiceServicer):
    def ProvideResult(self, request, context):
        print('Result:', parse_typed_value(request.typed_value))
        resp = results_pb2.ProvideResultResponse()
        resp.status = 0
        return resp
```

What is happening here? The `KVDataServer` just serves responses from the provided dict, and `TrivialResultServer` prints whatever it gets to stdout, after parsing it to `TypedValue` for readability. One remaining piece is to load the data from the `KVDataServer` to serve. Here is how we do it:

```python
def populate_value(out, val):
    sub_vals = val.get_sub_values()
    if sub_vals is None:
        out.bytes.data = bytes(val.get_bytes())        
        return
    for sub_val in sub_vals:
        cur = Value()
        populate_value(cur, sub_val)
        out.vector.value.append(cur)


def typed_val_to_proto(typed_val):
    val = TypedValue()
    val.type_json = typed_val.get_type().to_json_string()
    populate_value(val.value, typed_val.get_value())
    return val


def load_typed_values(path):
    with open(path, 'r') as f:
        json_data = f.read()
    typed_vals = [cc.TypedValue(json.dumps(item)) for item in json.loads(json_data)]
    vals = []
    for typed_val in typed_vals:        
        vals.append(typed_val_to_proto(typed_val)) 
    res = {'test'.encode(): vals}
    return res
```

This is a bit tricky: we read `TypedValue`'s, and we need to convert them to proto format (this conversion logic is not exported in the Python wrapper yet).

Now we have all the pieces to start data nodes. For each data node, we do the following:

```python
def read_file(path):
    return open(path, 'rb').read()


server_credentials = grpc.ssl_server_credentials(((
    read_file(os.path.join(getattr(args, 'data_dir%d' % i), 'data_key.pem')),
    read_file(os.path.join(getattr(args, 'data_dir%d' % i), 'data_cert.pem')),
),))

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
values = load_values(os.path.join(getattr(args, 'data_dir%d' % i), 'input_share_proto.txt'))
data_pb2_grpc.add_DataManagerServiceServicer_to_server(KVDataServer(values), server)
server.add_secure_port('[::]:%d' % getattr(args, 'data_port%d' % i),
                        server_credentials)
server.start()
servers.append(server)

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
results_pb2_grpc.add_ResultProviderServiceServicer_to_server(TrivialResultServer(), server)
server.add_secure_port('[::]:%d' % getattr(args, 'result_port%d' % i),
                        server_credentials)
server.start()
servers.append(server)
```

The only nontrivial part here is the `server_credentials` variable. We need to use the private key of the server to start it.

The only remaining part is orchestrator. It connects to compute nodes with the following helper:

```python
def create_client_stub(addr, ca_cert):
    NUM_RETIRES  = 100
    delay = 1.0
    for attempt in range(NUM_RETIRES):
        try:
            channel_credential = grpc.ssl_channel_credentials(ca_cert)
            channel = grpc.secure_channel(addr, channel_credential,
                                          # Certificates were generated for localhost, so we make sure
                                          # that we check them against 'localhost' domain.
                                          # This should not be used in production.
                                          options=[('grpc.ssl_target_name_override', 'localhost')])
            stub = party_pb2_grpc.PartyServiceStub(channel)
            _ = stub.ListSessions(party_pb2.SessionListRequest())
            return stub
        except:
            if attempt == NUM_RETIRES - 1:
                raise
            print("Could not connect, retrying")
            time.sleep(delay)
            delay *= 1.1
            delay = min(delay, 30.0)
```

Two things to note here:
*  We're doing retries, to avoid race conditions in what is started first;
*  We use the `grpc.ssl_target_name_override` parameter, to circumvent TLS domain name check. This is hacky and exposes us to MitM-style attacks by the root certificate owner. This is good for testing, but should not be used in production.

Then, we connect to nodes using this helper, register the graph and start the computation session:

```python
cert = read_file(args.ca_pem)
print('Connecting to compute nodes')
stubs = [
    create_client_stub('localhost:%d' % getattr(args, 'compute_port%d' % i),
                        cert)
    for i in range(3)
]
print('Connected to compute nodes')
graph_id = str(uuid.uuid4())
context = read_file(args.context_path)
for stub in stubs:
    resp = stub.RegisterGraph(party_pb2.RegisterGraphRequest(graph_json=context, graph_id=graph_id))
    assert resp.result == 0
print('Registered graphs')
sess_id = str(uuid.uuid4())
for i, stub in enumerate(stubs):
    data_key = data_pb2.DataRequest(key=b'test')
    req = party_pb2.CreateSessionRequest(
        session_id=sess_id, graph_id=graph_id, party_id=i,
        data_settings=data_key.SerializeToString())
    for j in range(3):
        addr = party_pb2.Address(address='localhost', port=getattr(args, 'compute_port%d' % j))
        req.party_address.append(addr)
    resp = stub.CreateSession(req)
    assert resp.result == 0
print('Created sessions')
```

Finally, we need to wait for the session to finish, which we do by repeatedly querying the list of sessions, and checking if they're still running:

```python
while True:
    done = True
    for stub in stubs:
        resp = stub.ListSessions(party_pb2.SessionListRequest())
        # Note: statuses 0 and 1 correspond to "finished" and "failed" respectively.
        if any([s.status in [0, 1] for s in resp.session]):
            done = False
            break
    if done:
        break
    time.sleep(0.1)
print('Sessions done')
for stub in stubs:
    _ = stub.FinishSession(party_pb2.FinishSessionRequest(session_id=sess_id))
```

Note that once everything is done, we need to call `FinishSession` to free up the memory. It doesn't matter for this example, but might matter for more complicated flows (e.g. ML model training).

### Running everything

Now that we have the implementation, it is time to start the runtime. We assume that `$DATA_DIR` contains the data layout for the prepackaged runtime example, and `example/scripts/run_compute_nodes.sh` invokes three runtime compute nodes. Let's start the compute nodes:

```bash
WORK_DIR=$DATA_DIR sh runtime/example/scripts/run_compute_nodes.sh
```

Now compute nodes are running, but they're just idling, since no one told them what to do. Let's start our Python code which creates data nodes and then runs orchestrator for the example graph:

```bash
cp -R $DATA_DIR/certificates/* $DATA_DIR/data/
python runtime/example/python/combined_example.py \
    --data_dir0=${DATA_DIR}/data/party0 \
    --data_dir1=${DATA_DIR}/data/party1 \
    --data_dir2=${DATA_DIR}/data/party2 \
    --ca_pem=${DATA_DIR}/data/ca.pem \
    --context_path=${DATA_DIR}/data/graph.json
``` 

After some time, it creates all the servers and establishes the connection to the compute nodes and runs the computation session. The output should look similar to the following:

```
Connecting to compute nodes
Connected to compute nodes
Registered graphs
Created sessions
Result: {"kind":"scalar","type":"u32","value":2800266097}
Result: {"kind":"scalar","type":"u32","value":511}
Result: {"kind":"scalar","type":"u32","value":787611686}
Sessions done
```

Note that it prints results from all parties, but we reveal it only to one party, so it gets the correct one (511), while the rest get random.
