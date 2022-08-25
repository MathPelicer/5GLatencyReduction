import utilities

workload = utilities.createWorkload(10)

#utilities.createNormalDistribution()


devicesList = []

for device in workload:
    print("id: " + str(device.device_id) + " class of service: " + device.class_of_service)

#for device in networkData["UserDevices"]:
#    print(device.attrib.values())

