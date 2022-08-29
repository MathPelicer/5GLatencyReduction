import json
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as pl
import seaborn as sb


def getDevices():
    list_of_devices = []

    f = open('workload.json')
    data = json.load(f)
    
    for i in data:
        for t in data[i]:

            device = []
            device.append(i)
            device.append(t['device_id'])
            device.append(0 if t['class_of_service'] == 'standard' else 1)
            device.append(t['latency'][0])
            device.append(t['latency'][1])

            list_of_devices.append(device)
    
    devicesDf = pd.DataFrame(list_of_devices, columns=['Hour', 'Device_id', 'Class_of_service', 'Fog_latency', 'Cloud_latency'])
    
    return devicesDf


def KmeansImplantation():
    df = getDevices()

    X = np.array(df.drop(['Device_id', 'Hour'], axis = 1))

    kmeans = KMeans(n_clusters= 2, n_init = 50, max_iter = 100)
    kmeans.fit_predict(X)
    df['K-class'] = kmeans.labels_
    
    print(df)

KmeansImplantation()