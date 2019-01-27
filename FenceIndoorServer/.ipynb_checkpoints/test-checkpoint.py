#interfacce test

import logging
import commonEngine as com
import dbEngine as dao
import annEngine as ann
import numpy as np
import pandas as pd
import json

#test del training
def testTraining():
    #costruisce i dati
    X, Y = ann.makeDataFromDb()
    
    #addestra l'ann
    ann.buildAndFitAnn(X, Y)

#export db into csv
def exportDBIntoCsv():
    X, Y = ann.makeDataFromDb()
    np.savetxt("./tmp/X.csv", X, delimiter=",",fmt='%.0f')
    np.savetxt("./tmp/Y.csv", Y, delimiter=",",fmt='%.0f')

#import from csv anf json
def importData():
    dfX = pd.read_csv('./tmp/X.csv')
    dfY = pd.read_csv('./tmp/Y.csv')
    X = dfX.iloc[:, :].values.astype(np.float32)
    Y = dfY.iloc[:, :].values.astype(np.float32)
    wifiMapEncode = json.load(open("./tmp/wifiMap.json"))
    areaMapDecode = json.load(open("./tmp/areaMap.json"))
    print(X)
    print(Y)
    print(wifiMapEncode)
    print(areaMapDecode)
    
#main
if __name__ == '__main__':
    #testTraining()
    #exportDBIntoCsv()
    importData()