import utilities
import json


workload = json.loads(utilities.createWorkload(1, 2))

distribution = utilities.createNormalDistribution()
# for i in range(len(distribution)):
# if i % 1000 == 0:
#print(str(i) + ": " + str(distribution[i] * 1000))
print(workload)

for device in workload:
    print(device)
    # print("id: " + str(device["device_id"]) +
    #      " class of service: " + device["class_of_service"])
