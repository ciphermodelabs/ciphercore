import argparse
from concurrent import futures
import grpc
import json
import logging
import os
import time
import uuid

import numpy as np

from value_pb2 import TypedValue
import data_pb2
import data_pb2_grpc
import results_pb2
import results_pb2_grpc
import party_pb2
import party_pb2_grpc


def load_typed_values(path):
    with open(path, 'r') as f:
        json_data = f.read()
    json_bytes_arr = json.loads(json_data)
    vals = []
    for json_bytes in json_bytes_arr:
        val = TypedValue()
        val.ParseFromString(bytes(json_bytes))
        vals.append(val) 
    res = {'test'.encode(): vals}
    return res


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
            if attempt >= NUM_RETIRES - 1:
                raise
            print("Could not connect, retrying")
            time.sleep(delay)
            delay *= 1.1
            delay = min(delay, 30.0)


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


class TrivialResultServer(results_pb2_grpc.ResultProviderServiceServicer):
    def ProvideResult(self, request, context):
        print('Got result:', request)
        print('Decoded:', np.frombuffer(request.typed_value.value.bytes.data, dtype=np.int32))
        resp = results_pb2.ProvideResultResponse()
        resp.status = 0
        return resp


def read_file(path):
    return open(path, 'rb').read()


def run_orchestrator(args):
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
    while True:
        done = True
        for stub in stubs:
            resp = stub.ListSessions(party_pb2.SessionListRequest())
            if any([s.status in [0, 1] for s in resp.session]):
                done = False
                break
        if done:
            break
        time.sleep(0.1)
    print('Sessions done')
    for stub in stubs:
        _ = stub.FinishSession(party_pb2.FinishSessionRequest(session_id=sess_id))


def main():
    parser = argparse.ArgumentParser()
    for i in range(3):
        parser.add_argument('--data_port%d' % i,
                            nargs='?',
                            type=int,
                            default=4201 + i * 10,
                            help='Port for the data node #%d' % i)
        parser.add_argument('--result_port%d' % i,
                            nargs='?',
                            type=int,
                            default=4202 + i * 10,
                            help='Port for the results node #%d' % i)
        parser.add_argument('--compute_port%d' % i,
                            nargs='?',
                            type=int,
                            default=4203 + i * 10,
                            help='Port for compute #%d' % i)
        parser.add_argument('--data_dir%d' % i,
                            type=str,
                            default='',
                            help='Path to the data for party #%d' % i)
    parser.add_argument('--ca_pem',
                        type=str,
                        default='',
                        help='Path to the CA certificate')
    parser.add_argument('--context_path',
                        type=str,
                        default='',
                        help='Path to the MPC-compiled context')
    args = parser.parse_args()

    servers = []
    # Data and result nodes.
    for i in range(3):
        server_credentials = grpc.ssl_server_credentials(((
            read_file(os.path.join(getattr(args, 'data_dir%d' % i), 'data_key.pem')),
            read_file(os.path.join(getattr(args, 'data_dir%d' % i), 'data_cert.pem')),
        ),))

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        typed_values = load_typed_values(os.path.join(getattr(args, 'data_dir%d' % i), 'input_share_proto.txt'))
        data_pb2_grpc.add_DataManagerServiceServicer_to_server(KVDataServer(typed_values), server)
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

    # Let the potato rest for N minutes.
    # Note: this is happening because of the chicken-and-egg problem: compute nodes want to connect to
    # data nodes before opening their port, and we don't have retries here.
    time.sleep(15)

    # Actually run the stuff.
    run_orchestrator(args)

    for server in servers:
        server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    main()
