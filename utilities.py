import json
from importlib_metadata import distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from uuid import uuid4
from random import randint

# basic access point to be extended


class ProcessingNode(object):
    used_bandwidth = 0
    available_bandwidth = 0
    used_processing = 0
    available_processing = 0
    connected_devices = []

    def __init__(self):
        pass


class CloudNode(ProcessingNode):
    device_capacity = 100

    def __init__(self):
        pass


class FogNode(ProcessingNode):
    device_capacity = 20

    def __init__(self):
        pass


class Devices():
    device_id = 0
    # fog latency = 98
    # cloud latency = 196
    latency = [98, 196]

    def __init__(self, class_of_service):
        self.class_of_service = class_of_service  # priority/standard
        self.device_id = str(uuid4())
        self.latency = self.latency
        self.request_size = randint(50, 1000)


def createNormalDistribution():
    data = np.arange(0, 24, 0.001)
    pdf = norm.pdf(data, loc=12, scale=4)
    #pdf = 23000
    #plt.plot(data, pdf, color='black')
    # plt.show()
    times_distribution = []

    for i in range(len(pdf)):
        if i % 1000 == 0:
            times_distribution.append(pdf[i] * 10)

    return times_distribution


def createWorkload(number_cloud_nodes, number_fog_nodes):
    distribution = createNormalDistribution()

    workload = {}

    for i in range(len(distribution)):
        size = round(((number_cloud_nodes * 100) +
                      (number_fog_nodes * 20)) * distribution[i])

        for device in range(size):
            priority = randint(0, 1)
            if priority == 0:
                new_device = Devices("standard")
            else:
                new_device = Devices("priority")

            if i in workload:
                workload[i].append(new_device.__dict__)
            else:
                workload[i] = []
                workload[i].append(new_device.__dict__)

    print(workload)
    out_file = open("workload.json",  "w")
    json_workload = json.dump(workload, out_file, indent=2)
    out_file.close()
    return json_workload
