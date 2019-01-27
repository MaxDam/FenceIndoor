#interfacce test

import logging
import commonEngine as com
import dbEngine as dao
import annEngine as ann
import numpy as np

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

#main
if __name__ == '__main__':
    #testTraining()
    exportDBIntoCsv()