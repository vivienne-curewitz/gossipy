def node_process(local_ip, base_port, i, report_queue: Queue):
    # print(f"Start Node at {local_ip}:{base_port+i}")
    if False: #i%10 == 0:
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

def sample_mean(base_port: int, max_id: int):
    while True:
        port = randint(base_port, base_port+max_id)
        try:
            resp = requests.get(f"http://0.0.0.0:{port}/inference", timeout=0.25)
            jdata = resp.json()
            print(f"Sampled mean from node {port}: {jdata['inference']}")
        except Exception:
            print(f"Failed to connect to node {port}")
            pass
        time.sleep(2)

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
        time.sleep(0.05)
    sample_thread = Thread(target=sample_mean, args=[base_port, 100], daemon=True)
    # time.sleep(10)
    sample_thread.start()
    # plot stuff here: start a background thread that consumes report_queue
    def _plotter_loop(inqueue, stream_ids):
        plot = StreamPlot(inqueue, stream_ids)
        while True:
            try:
                plot.plot_graph()
            except Exception:
                pass
            time.sleep(0.25)

    reporter_ids = [i+1 for i in range(gns-1) if (i+1)%10 == 0]
    # plot_thread = Thread(target=_plotter_loop, args=(report_queue, reporter_ids), daemon=True)
    # plot_thread.start()
    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received: terminating child processes...")
        # Terminate any still-running child processes
        for proc in procs:
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:
                pass
        # Give them a moment to exit
        time.sleep(0.5)
        for proc in procs:
            try:
                proc.join(timeout=1)
            except Exception:
                pass
        print("All child processes terminated. Exiting.")


if __name__ == "__main__":
    # gn = GossipNode("base", "0.0.0.0", 8080)
    # gn.start()
    run_many(101)