import utilities

networkData = utilities.xmlParser('network.xml')
utilities.createNormalDistribution()
print(networkData)

devicesList = []
grid_map = utilities.Grid

for device in networkData["UserDevices"]:
    if(grid_map.IsInsideGrid(grid_map, device.attrib["pos_x"], device.attrib["pos_y"])):
        grid_map.devices_list.append(device)
    else:
        print("position outside threshold")
    
    #print(device)
    #print(device.attrib.values())

