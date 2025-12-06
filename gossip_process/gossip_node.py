# this should be run in a process. It maintains three threads:
# 1. Heavy processing (SVM, CNN, KMeans, etc)
# 2. Listener - waits for gossip from a friend
# 3. Chatter - sends gossip to a friend
# obviously they aren't friends, just sister processes which form the nodes that are walked by the learning algorithm.
from threading import Thread, Queue
import requests
from random import randint
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

# god python is so bad at networking; wtf
class PeerHandler(BaseHTTPRequestHandler):
    parent = None

    def __init__(self, parent):
        self.parent = parent
    def do_GET(self):
        if self.path == "/ping":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"pong")
        elif self.path == "/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = {"ok": True, "id": "peer123"}
            self.wfile.write(json.dumps(payload).encode())
        else:
            self.send_response(404)
            self.end_headers()

class GossipNode:
    name: str
    ip: str
    port: int
    models: list
    chatter: Thread
    worker: Thread
    inqueue: Queue
    outqueue: Queue
    max_peers: int
    num_peers_to_share: int
    peers: list[str] 
    def __init__(self, name: str, ip: str, port: int, max_peers=10, num_peers_to_share = 5):
        self.name = name
        self.ip = ip
        self.port = port
        self.inqueue = Queue()
        self.outqueue = Queue()
        self.max_peers = max_peers
        self.num_peers_to_share = num_peers_to_share
    
    def listen(self):
        ph = PeerHandler(self)
        server = HTTPServer((self.url, self.port), ph)
        print("Server running")
        server.serve_forever()

    def send(self):
        model_params = self.outqueue.get()
        num_peers = len(self.peers)
        peer = self.peers[self.peers[randint(num_peers)]]
        if num_peers > self.num_peers_to_share:
            send_peers = [randint(num_peers) for x in range(max(self.num_peers_to_share))]
        else:
            send_peers = self.peers
        send_data = {
            "peers": send_peers,
            "model": model_params
        }
        requests.put(f"{peer}new", json.dumps(send_data))