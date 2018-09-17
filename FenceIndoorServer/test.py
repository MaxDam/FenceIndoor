#interfacce test

import logging
import commonEngine as com
import dbEngine as dao
import annEngine as ann

#test del training
def testTraining():
    #costruisce i dati
    X, Y = ann.makeDataFromDb()
    
    #addestra l'ann
    ann.buildAndFitAnn(X, Y)

#main
if __name__ == '__main__':
    testTraining()