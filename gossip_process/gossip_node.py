# this should be run in a process. It maintains three threads:
# 1. Heavy processing (SVM, CNN, KMeans, etc)
# 2. Listener - waits for gossip from a friend
# 3. Chatter - sends gossip to a friend
# obviously they aren't friends, just sister processes which form the nodes that are walked by the learning algorithm.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from threading import Thread 
from queue import Queue, Empty
import requests
from random import randint
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from multiprocessing import Process, Queue as MPQueue
from worker import MeanWorker, ReportingMeanWorker
from utils.graph_process import plot_process
from urllib3.exceptions import MaxRetryError
# god python is so bad at networking; wtf
class PeerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return
        # if "code" in format:
            # return  # ignore "GET /path -> 200" type logs
        # super().log_message(format, *args)

    def do_POST(self):
        if self.path == "/peers":
            # parse
            clength = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(clength)
            # print(f"Got peer request: {str(data)}")
            jdata = json.loads(data)
            self.server.node.add_peer(jdata["address"])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"peers": self.server.node.peers}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_PUT(self):
        if self.path == "/data":
            self.send_response(200)
            self.end_headers()
            clength = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(clength)
            self.server.node.handle_data_put(data)
        else:
            self.send_response(404)
            self.end_headers()

class PeerHttpServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, node):
        super().__init__(server_address, RequestHandlerClass)
        self.node = node  # <--- shared state

class GossipNode:
    name: str
    ip: str
    port: int
    models: list
    inqueue: Queue
    outqueue: Queue
    max_peers: int
    worker: MeanWorker
    # num_peers_to_share: int
    peers: list[str] 
    def __init__(self, name: str, ip: str, port: int, worker, max_peers=10):
        self.name = name
        self.ip = ip
        self.port = port
        self.inqueue = Queue()
        self.outqueue = Queue()
        self.max_peers = max_peers
        # self.num_peers_to_share = num_peers_to_share
        self.peers = []
        self.worker = worker
        self.worker.inqueue = self.inqueue
        self.worker.outqueue = self.outqueue

    def start(self):
        # start other threads
        chatter = Thread(target=self.send, daemon=True) 
        worker_thread = Thread(target=self.worker.run, daemon=True)
        listener_thread = Thread(target=self.listen, daemon=True)
        # start listen thread
        listener_thread.start()
        # a bit hacky, but this is the case when it has been seeded with a base peer
        if len(self.peers) == 1:
            self.get_peers()
        chatter.start()
        worker_thread.start()
        # stay alive
        listener_thread.join()

    def get_peers(self):
        try:
            resp = requests.post(f"http://{self.peers[0]}/peers", json={"address": f"{self.ip}:{self.port}"}, timeout=10)
        except MaxRetryError:
            print(f"Node {self.name} failed to get peers from {self.peers[0]}")
            return
        jdata = resp.json()
        self.add_many_peers(jdata["peers"])
    
    # always adds the peer
    def add_peer(self, peer):
        if peer in self.peers:
            return
        if len(self.peers) >= self.max_peers:
            # random replace
            self.peers[randint(0, len(self.peers) - 1)] = peer
        else:
            self.peers.append(peer)

    def add_many_peers(self, peers: list[str]):
        self.peers += peers
        # i know this sucks
        self.peers = list(set(self.peers))
        while len(self.peers) > self.max_peers:
           rem_ind = randint(0, len(self.peers) - 1) 
           self.peers.pop(rem_ind)

    def listen(self):
        server = PeerHttpServer((self.ip, self.port), PeerHandler, self)
        # print("Server running")
        server.serve_forever()

    def get_random_peer(self):
        if len(self.peers) < 1:
            return None
        if len(self.peers) == 1:
            return self.peers[0]
        pi = randint(0, len(self.peers) - 1)
        return self.peers[pi]

    def send(self):
        while True:
            try:
                model_params = self.outqueue.get(timeout=1)
            except Empty as e:
                model_params = self.worker.data
            # print(f"{self.name} -- Peers: {self.peers}")
            peer = self.get_random_peer()
            if peer is None:
                return
            send_data = {
                "peers": self.peers,
                "model": model_params
            }
            requests.put(f"http://{peer}/data", json.dumps(send_data))

    def handle_data_put(self, data):
        # data is encode json
        jdata = json.loads(data)
        peers = jdata["peers"]
        model = jdata["model"]
        self.add_many_peers(peers)
        self.inqueue.put(model)

def node_process(local_ip, base_port, i, report_queue: Queue):
    # print(f"Start Node at {local_ip}:{base_port+i}")
    if i%10 == 0:
        wt = ReportingMeanWorker(i, report_queue, i)
    else:  
        wt = MeanWorker(i)
    gt = GossipNode(f"gp{i}", local_ip, base_port+i, wt)
    gt.add_peer(f"{local_ip}:{base_port}")
    gt.start()

def node_base_process(local_ip, base_port, i):
    # print(f"Start Node at {local_ip}:{base_port+i}")
    wt = MeanWorker(i)
    gt = GossipNode(f"gp{i}", local_ip, base_port+i, wt)
    gt.start()

def run_many(gns: int):
    procs: list[Process] = []
    base_port = 8080
    local_ip = "0.0.0.0"
    w0 = MeanWorker(0)
    report_queue = MPQueue(1000)
    # gn.start()
    procs.append(Process(target=node_base_process, args=[local_ip, base_port, 0], daemon=True))
    for i in range(gns - 1):
        procs.append(Process(target=node_process, args=[local_ip, base_port, i+1, report_queue], daemon=True))
    for proc in procs:
        proc.start()
    # plot stuff here
    plot_process(report_queue, [i+1 for i in range(gns-1) if (i+1)%10 == 0])
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    # gn = GossipNode("base", "0.0.0.0", 8080)
    # gn.start()
    run_many(1)