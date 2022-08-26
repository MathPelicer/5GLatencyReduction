import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from uuid import uuid4
from random import randint

# basic access point to be extended
class ProcessingNode(object):
    connected_devices = []
    used_bandwidth = 0
    available_bandwidth = 0
    used_processing = 0
    available_processing = 0

    def __init__(self):
        pass

class CloudNode(ProcessingNode):
    def __init__(self):
        pass

class FogNode(ProcessingNode):
    def __init__(self):
        pass

class Devices():
    device_id = 0
    # fog latency = 98
    # cloud latency = 196
    latency = [98, 196]
    
    def __init__(self, class_of_service):
        self.class_of_service = class_of_service # priority/standard
        self.device_id = str(uuid4())
        self.latency = self.latency

def createNormalDistribution():
    data = np.arange(0, 24, 0.001)
    pdf = norm.pdf(data, loc=12, scale=4)
    #pdf = 23000
    plt.plot(data, pdf, color='black')
    plt.show()

    return pdf

def createWorkload(size):
    workload = []

    for device in range(size):
        priority = randint(0, 1)
        if priority == 0:
            new_device = Devices("standard")
        else:
            new_device = Devices("priority")
        workload.append(new_device)

    json_workload = json.dumps([dev.__dict__ for dev in workload])
    return json_workload