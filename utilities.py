import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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
    classOfService = "standard/priority"
    latency = []
    
    def __init__(self):
        pass

def xmlParser(xmlFile):
	#keep the configuration parameters
	parameters = {}
	#construct the XML tree
	tree = ET.parse(xmlFile)
	#get the root
	root = tree.getroot()
	#return root
	##iterate over the nodes and  store each one into the parameters dictionaire
	for child in root:
		parameters[child.tag] = child
	return parameters

def createNormalDistribution():
    data = np.arange(1, 24, 0.001)
    pdf = norm.pdf(data, loc=12, scale=4)
    #pdf = 23000
    plt.plot(data, pdf, color='black')
    plt.show()

def createWorkload(size):
    pass
