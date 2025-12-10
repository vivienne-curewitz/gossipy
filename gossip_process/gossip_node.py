# this should be run in a process. It maintains three threads:
# 1. Heavy processing (SVM, CNN, KMeans, etc)
# 2. Listener - waits for gossip from a friend
# 3. Chatter - sends gossip to a friend
# obviously they aren't friends, just sister processes which form the nodes that are walked by the learning algorithm.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import aiohttp
from aiohttp import web
from threading import Thread
from queue import Queue, Empty
import requests
from random import randint
import json
from multiprocessing import Process, Queue as MPQueue
# from worker import MeanWorker, ReportingMeanWorker
from utils.graph_process import StreamPlot
from urllib3.exceptions import MaxRetryError
import time
# god python is so bad at networking; wtf
# The HTTP listener and sender are implemented with aiohttp (async).
# The worker remains a background thread and communicates via thread-safe queues.

class GossipNode:
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
        # Start the worker thread (keeps heavy work in a background thread)
        worker_thread = Thread(target=self.worker.run, daemon=True)
        worker_thread.start()

        # If seeded with a base peer, fetch peers synchronously before starting async server
        if len(self.peers) == 1:
            self.get_peers()

        # Run aiohttp server and async sender in the asyncio event loop (blocks)
        try:
            asyncio.run(self._run_async_components())
        except KeyboardInterrupt:
            pass

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
        # Deprecated: listener is now implemented with aiohttp in _run_async_components
        raise RuntimeError("listen() is deprecated; use async server")

    def get_random_peer(self):
        if len(self.peers) < 1:
            return None
        if len(self.peers) == 1:
            return self.peers[0]
        pi = randint(0, len(self.peers) - 1)
        return self.peers[pi]

    def send(self):
        # Deprecated: sending is now async in _sender_loop
        raise RuntimeError("send() is deprecated; use async sender")

    def handle_data_put(self, data):
        # Synchronous helper for tests or backwards compatibility
        jdata = json.loads(data)
        peers = jdata.get("peers", [])
        model = jdata.get("model")
        self.add_many_peers(peers)
        if model is not None:
            self.inqueue.put(model)

    # --- Async server / sender implementation ---
    async def _handle_peers(self, request):
        payload = await request.json()
        self.add_peer(payload.get("address"))
        return web.json_response({"peers": self.peers})

    async def _handle_data(self, request):
        payload = await request.json()
        self.add_many_peers(payload.get("peers", []))
        model = payload.get("model")
        if model is not None:
            # thread-safe put
            self.inqueue.put(model)
        return web.Response(status=200)

    async def _handle_mean(self, request):
        result = self.worker.current_mean
        return web.json_response({"inference": result})

    async def _handle_inference(self, request):
        payload = await request.json()
        data = payload.get("data")
        if data is None:
            return web.Response(status=400, text="Missing data")
        result = self.worker.infer(data)
        return web.json_response({"inference": result})

    async def _start_aio_server(self):
        app = web.Application()
        app.router.add_post("/peers", self._handle_peers)
        app.router.add_put("/data", self._handle_data)
        app.router.add_get("/mean", self._handle_mean)
        app.router.add_post("/inference", self._handle_inference)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.ip, self.port)
        await self._site.start()

    async def _stop_aio_server(self):
        if hasattr(self, "_runner") and self._runner:
            await self._runner.cleanup()

    async def _sender_loop(self):
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                try:
                    model_params = await asyncio.get_running_loop().run_in_executor(
                        None, self.outqueue.get, True, 0.5
                    )
                except Empty:
                    model_params = getattr(self.worker, "model", None)

                peer = self.get_random_peer()
                if peer is not None and model_params is not None:
                    send_data = {"peers": self.peers, "model": model_params}
                    try:
                        resp = await session.put(f"http://{peer}/data", json=send_data)
                        if resp.status != 200:
                            print(f"Node {self.name} failed to send data to {peer} with status {resp.status}")
                    except Exception as e:
                        print(f"Node {self.name} failed to send data to {peer} with error {e}")
                        # ignore send errors
                        pass
                else:
                    if peer is None:
                        print(f"Node {self.name} has no peers to send to.")
                    else:
                        print(f"Node {self.name} has no model to send.")

    async def _run_async_components(self):
        # start server and sender
        await self._start_aio_server()
        send_task = asyncio.create_task(self._sender_loop())

        try:
            # run forever until cancelled
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        finally:
            send_task.cancel()
            await self._stop_aio_server()

