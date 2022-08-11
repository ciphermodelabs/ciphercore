import argparse
import ciphercore as cc
from concurrent import futures
import grpc
import json
import logging
import os
import time
import uuid

import numpy as np

from value_pb2 import TypedValue, Value
import data_pb2
import data_pb2_grpc
import results_pb2
import results_pb2_grpc
import party_pb2
import party_pb2_grpc


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
        typed_values = load_typed_values(os.path.join(getattr(args, 'data_dir%d' % i), 'input.txt'))
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
