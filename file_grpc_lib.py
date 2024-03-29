import os
from concurrent import futures

import grpc
import time

import filetrans_pb2 as pb2
import filetrans_pb2_grpc as pb2_grpc
import random,string
# CHUNK_SIZE = 1024 * 1024  # 1MB
CHUNK_SIZE = 2154387


def get_file_chunks(filename):
    with open(filename, 'rb') as f:
        while True:
            piece = f.read(CHUNK_SIZE);
            if len(piece) == 0:
                return
            yield pb2.Chunk(buffer=piece)


def save_chunks_to_file(chunks, filename):
    with open(filename, 'wb') as f:
        for chunk in chunks:
            f.write(chunk.buffer)


class FileClient:
    def __init__(self, address):
        channel = grpc.insecure_channel("10.10.1.3:9991")
        self.stub = pb2_grpc.FileServerStub(channel)

    def upload(self, in_file_name):
        chunks_generator = get_file_chunks(in_file_name)
        response = self.stub.upload(chunks_generator)
        assert response.length == os.path.getsize(in_file_name)

    def download(self, target_name, out_file_name):
        response = self.stub.download(pb2.Request(name=target_name))
        save_chunks_to_file(response, out_file_name)


class FileServer(pb2_grpc.FileServerServicer):
    def __init__(self):

        class Servicer(pb2_grpc.FileServerServicer):
            def __init__(self):
                self.tmp_file_name = ''
                # letters = string.ascii_lowercase
                # ''.join(random.choice(letters) for i in range(10))

            def upload(self, request_iterator, context):
                save_chunks_to_file(request_iterator, self.tmp_file_name)
                return pb2.Reply(length=os.path.getsize(self.tmp_file_name))

            def download(self, request, context):
                if request.name:
                    return get_file_chunks(self.tmp_file_name)

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        pb2_grpc.add_FileServerServicer_to_server(Servicer(), self.server)

    def start(self):
        self.tmp_file_name = ''
        # self.server.add_insecure_port(f'[::]:{port}')
        self.server.add_insecure_port("10.10.1.3:9991")
        self.server.start()

        try:
            while True:
                time.sleep(10)
                # time.sleep(60*60*24)
        except KeyboardInterrupt:
            self.server.stop(0)
    
    def stop_me(self):
        self.server.stop(0)
