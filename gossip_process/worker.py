from multiprocessing import Queue as MPQueue
from queue import Queue
import time

class MeanWorker:
    inqueue: Queue
    outqueue: Queue
    my_val: int
    current_mean: float

    def __init__(self, value: float):
        self.my_val = value
        self.current_mean = self.my_val

    @property
    def data(self):
        return self.current_mean

    def run(self):
        while True:
            next_val = self.inqueue.get()
            self.current_mean = (self.current_mean + next_val)/2
            self.outqueue.put(self.current_mean)
            time.sleep(0.1)


class ReportingMeanWorker(MeanWorker):
    report_queue: MPQueue
    id: int

    def __init__(self, value: float, report_queue: MPQueue, id: int):
        super().__init__(value)
        self.report_queue = report_queue
        self.id = id
        print(f"ReportingMeanWorker {self.id} created")

    def run(self):
        while True:
            next_val = self.inqueue.get()
            self.current_mean = (self.current_mean + next_val)/2
            self.outqueue.put(self.current_mean)
            self.report_queue.put((self.id, self.current_mean))
            # print(f"Mean = {self.current_mean}")
            time.sleep(0.1)

    