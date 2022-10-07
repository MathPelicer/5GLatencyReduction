import json
from importlib_metadata import distribution
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from uuid import uuid4
from random import randint, uniform

# basic access point to be extended


class ProcessingNode(object):
    used_bandwidth = 0
    available_bandwidth = 0
    used_processing = 0
    available_processing = 0

    def __init__(self):
        pass


class CloudNode(ProcessingNode):
    device_capacity = 1000

    def __init__(self):
        self.connected_devices = []
        self.queue_devices = []


class FogNode(ProcessingNode):
    device_capacity = 200

    def __init__(self):
        self.connected_devices = []
        self.queue_devices = []


class Devices():
    device_id = 0
    # fog latency = 98
    # cloud latency = 196

    def __init__(self, class_of_service, region):
        self.class_of_service = class_of_service  # priority/standard
        self.device_id = str(uuid4())
        self.latency = []
        self.request_size = randint(50, 1000)
        self.region = region

    def getListLatencyToAllFogNodes(self, number_fog_nodes, device_region):
        latency_list = []
        FOG_LATENCY = 98
        CLOUD_LATENCY = 196

        for region_fog_node in range(1, number_fog_nodes + 1):
            distance_to_fog_node = device_region - (region_fog_node - 0.5)
            normalized_to_latency = FOG_LATENCY * distance_to_fog_node

            latency_list.append(abs(normalized_to_latency))

        latency_list.append(CLOUD_LATENCY)

        return latency_list


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


def instantiate_processing_nodes(number_cloud_nodes, number_fog_nodes):
    processing_nodes = []

    for fog_nodes in range(number_fog_nodes):
        new_fog = FogNode()
        processing_nodes.append(new_fog)

    for cloud_nodes in range(number_cloud_nodes):
        new_cloud = CloudNode()
        processing_nodes.append(new_cloud)

    return processing_nodes


def createWorkload(number_cloud_nodes, number_fog_nodes):
    distribution = createNormalDistribution()
    region_range = number_fog_nodes

    workload = {}

    for i in range(len(distribution)):
        size = round(((number_cloud_nodes * 1000) +
                      (number_fog_nodes * 200)) * distribution[i])

        for device in range(size):
            region = uniform(0, region_range)
            priority = randint(0, 1)

            if priority == 0:
                new_device = Devices("standard", region)
            else:
                new_device = Devices("priority", region)

            new_device.latency = new_device.getListLatencyToAllFogNodes(
                region_range, region)

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


def createWorkload2(number_cloud_nodes, number_fog_nodes):
    distribution = createNormalDistribution()
    region_range = number_fog_nodes

    workload = {}

    for i in range(len(distribution)):
        size = round(((number_cloud_nodes * 1000) +
                      (number_fog_nodes * 200)) * distribution[i])

        # fog nodes have 37.5% of the capacity of the network
        # cloud node have 62.5% of the capacity
        #########################################################################
        # TODO:
        # Make this function generate the workload following this proportion
        #######################################################################
        for device in range(size):
            region = uniform(0, region_range)
            priority = randint(0, 1)

            if priority == 0:
                new_device = Devices("standard", region)
            else:
                new_device = Devices("priority", region)

            new_device.latency = new_device.getListLatencyToAllFogNodes(
                region_range, region)

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
