import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class Grid(object):
    size_x = 10
    size_y = 10
    access_points_list = []
    devices_list = []

    def __init__(self):
        pass

    def IsInsideGrid(self, pos_x, pos_y):
        if int(pos_x) <= self.size_x and int(pos_y) <= self.size_y and int(pos_x) >= 0 and int(pos_y) >= 0:
            return True
        else:
            return False

# basic access point to be extended
class AccessPoint(object):
    connected_devices = []
    used_bandwidth = 0
    available_bandwidth = 0
    pos_x = 0
    pos_y = 0

    def __init__(self):
        pass

class CloudAccessPoint(AccessPoint):
    def __init__(self):
        pass

class FogAccessPoint(AccessPoint):
    def __init__(self):
        pass

class Devices():
    device_id = 0
    pos_x = 0
    pos_y = 0
    application_type = "standard/priority"
    latency = 0
    
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
