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
    #ae = ann2.VarationAutoencoder(10)
    #ae = ann2.VarationAutoencoder2(10)
    fc = ann2.FullyConnectionLayer()
    ae.verbose=0
    fc.verbose=0
    fc.use_dropout=False   
    #X, Y = getData.getTrainingData()
    X, Y = getData.getTrainingDataFromFile()
    scoreAE = ae.buildAndFit(X)
    X = ae.predict(X)
    scoreANN = fc.buildAndFit(X, Y)
    ae.plotTrain()
    print('AE test score:', scoreAE[0])
    print('AE test accuracy:', scoreAE[1])
    fc.plotTrain()
    print('ANN test score:', scoreANN[0])
    print('ANN test accuracy:', scoreANN[1])

    