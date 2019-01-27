#interfacce test

import logging
import commonEngine as com
import dbEngine as dao
import annEngine as ann
import numpy as np
import pandas as pd
import json
import annEngine2 as ann2

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
    #importData()
    getData = ann2.GetData()
    ae = ann2.Autoencoder()
    fc = ann2.FullyConnectionLayer()
    ae.verbose=0
    fc.verbose=0
    fc.use_dropout=False
    X, Y = getData.getTrainingData()
    score1 = ae.buildAndFit(X)
    X = ae.predict(X)
    score2 = fc.buildAndFit(X, Y)
    print('AE test score:', score1[0])
    print('AE test accuracy:', score1[1])
    print('ANN test score:', score2[0])
    print('ANN test accuracy:', score2[1])
    ae.plotTrain()
    fc.plotTrain()

    