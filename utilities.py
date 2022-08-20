import xml.etree.ElementTree as ET

class grid(object):
    size_x = 0
    size_y = 0
    access_points_list = []
    devices_list = []

    def __init__(self):
        pass

# basic access point to be extended
class AccessPoint(object):
    connected_devices = []
    used_bandwidth = 0
    available_bandwidth = 0
    pos_x = 0
    pos_y = 0

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