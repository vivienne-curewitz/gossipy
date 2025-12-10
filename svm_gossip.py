from gossip_process.gossip_node import GossipNode
from gossip_process.svm_worker import SVMWorker, scale_df
import pandas as pd
from multiprocessing import Process
import time
import requests
from random import randint
from sklearn.metrics import accuracy_score

# init data set and split into trainging and testing here
def init_data(filename):
    df = pd.read_csv(filename)
    df_scaled = scale_df(df)
    train_df = df_scaled.sample(frac=0.4, random_state=42)
    test_df = df_scaled.drop(train_df.index)
    return train_df, test_df

def svm_gossip_process(local_ip, base_port, i, data):
    # init is a bit messy, should be refactored later
    node = GossipNode(
        f"svm_gp{i}",
        local_ip,
        base_port + i,
        SVMWorker(data, inqueue=None, outqueue=None),
    )
    if i > 0:
        node.add_peer(f"{local_ip}:{base_port}")
    node.start()

def test_accuracy(num_nodes, test_df, base_port):
    hit = 0
    samples = 100
    sample = test_df.sample(n=samples)
    x_test = sample.iloc[:, :-1].values
    y_test = sample.iloc[:, -1].values
    i = randint(0, num_nodes - 1)
    port = base_port + i
    try:
        data = {
            "data": x_test.tolist()
        }
        resp = requests.post(f"http://127.0.0.1:{port}/inference", json=data, timeout=1)
        if resp.ok:
            jdata = resp.json()
            preds = jdata["inference"]
            acc = accuracy_score(preds, y_test)
            print(f"\nAccuracy sampled from node {port}: {acc}")
        else:
            print(f"Node {port} returned error status {resp.status_code}")
    except Exception as e:
        # print(f"Failed to connect to node {port} with error {e}")
        pass
    

def run_n_nodes(local_ip, base_port, n_nodes, filename):
    train_df, test_df = init_data(filename)
    size = max(1, int(len(train_df) / n_nodes))
    procs = []
    for i in range(n_nodes):
        procs.append(Process(target=svm_gossip_process, args=[local_ip, base_port, i, train_df.sample(n=size)], daemon=True))
    for proc in procs:
        proc.start()
        # time.sleep(0.05)
    j = 0
    print("Awaiting Process Start")
    # time.sleep(5)
    while True:
        test_accuracy(n_nodes, test_df, base_port)
        j += 1
        # time.sleep(1)  

if __name__ == "__main__":
    print("Running gossip SVM test...")
    run_n_nodes("0.0.0.0", 8080, 20, "data/spambase.csv")