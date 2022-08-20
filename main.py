import utilities

networkData = utilities.xmlParser('network.xml')
print(networkData)

devicesList = []

for device in networkData["UserDevices"]:
    print(device)
    print(device.attrib.values())