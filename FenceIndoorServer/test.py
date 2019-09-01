#import common as com
#import dataLayer as dl
#import neuralNetwork as ann
import numpy as np

'''
#json inviato al body della request sendData
[
    [
        {
            'wifiLevel': 7, 
            'wifiName': 'Vodafone-34191642', 
            'serialVersionUID': 9220682644482277707
        }, 
        {
            'wifiLevel': 9, 
            'wifiName': 'Telecom-58058879', 
            'serialVersionUID': 9220682644482277707
        }
    ],
    [
        {
            'wifiLevel': 6, 
            'wifiName': 'Vodafone-34191642', 
            'serialVersionUID': 9220682644482277708
        }, 
        {
            'wifiLevel': 10, 
            'wifiName': 'Telecom-58058879', 
            'serialVersionUID': 9220682644482277708
        }
    ]
]
'''

'''
#json inviato al metodo predict
[
    [
        {
            'wifiLevel': 8, 
            'wifiName': 'Vodafone-34191642', 
            'serialVersionUID': 9220682644482277707
        }, 
        {
            'wifiLevel': 8, 
            'wifiName': 'Telecom-58058879', 
            'serialVersionUID': 9220682644482277707
        }
    ], 
    [
        {
            'wifiLevel': 9, 
            'wifiName': 'Vodafone-34191642', 
            'serialVersionUID': 9220682644482277708
        }, 
        {
            'wifiLevel': 8, 
            'wifiName': 'Telecom-58058879', 
            'serialVersionUID': 9220682644482277708
        }
    ]
]
'''

'''
print()
print("lista di wifi dal db")
wifiList = dl.getWifiListFromDb()
for i in range(10):
    print(wifiList[i])
print("...")
'''

'''
print()
print("lista di aree dal db")
areaList = dl.getAreaListFromDb()
for area in areaList:
    print(area)
'''

'''
print()
print("lista coppie area-scanId")
areaScanList = dl.getAreaAndScanIdListFromDb()
for i in range(10):
    print(areaScanList[i])
print("...")
'''

'''
print()
areaScan = areaScanList[0]
areaName = areaScan["area"]
scanId = areaScan["scanId"]
print("scansioni per", areaName, " e ", scanId, ":")
scanList = dl.getScansFromDb(areaName, scanId)
for i in range(10):
    print(scanList[i])
print("...")
'''


print()
print("prediction matrix di esempio:")
outputPredictMatrix = np.array([
    [0.9,0.1],
    [0.7,0.3],
    [0.4,0.6]
])
print(outputPredictMatrix)


'''
print()
print("vettore con somma di tutti gli output:")
sumOutputPredictVector = np.sum(outputPredictMatrix, axis=0)
print(sumOutputPredictVector)
'''

'''
print()
print("indice della classe con la massima previsione:")
maxPredictionIndex = np.argmax(sumOutputPredictVector)
print(maxPredictionIndex)
'''