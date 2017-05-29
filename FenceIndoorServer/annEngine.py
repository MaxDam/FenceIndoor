
#funzioni di utilita' ANN e di preparazione dati ETL

import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
#import petl as etl
import math
import dbEngine as dao


#classificatore rete neurale artificiale
classifier = None

#numero totale delle reti wifi
wifiCount = 0 

#numero totale delle aree
areaCount = 0

#numero totale delle scansioni
scanCount = 0

#una dictionary ogni <wifiName> con un numero sequenziale rappresentante la colonna della matrice di input
wifiMap = {}
  
#una dictionary che associa il nome dell'area con un numero sequenziale rappresentante la colonna della matrice di output
areaMap = {}
  

#costruisce la matrice di input ed il vettore di output 
#in base ai dati presenti sul db
def makeDataFromDb():
    
    global wifiCount, wifiMap, areaCount, scanCount, areaMap
    
    #step 1 
    #ottiene l'elenco delle wifi acquisite  nelle scansioni, e le itera per:
    # - assegnare a <wifiCount> il numero totale delle wifi acquisite
    # - associare con <wifiMap> ogni <wifiName> con un numero sequenziale rappresentante la colonna della matrice di input
    wifiList = dao.getWifiListFromDb()
    for wifi in wifiList:
        wifiName = wifi['wifiName']
        wifiMap[wifiName] = wifiCount
        wifiCount = wifiCount + 1
        
    #step 2
    #ottiene le aree dal database, e le itera per:
    # - assegnare a <scanCount> la sommatoria dei <lastScanId> delle aree
    # - assegnare ad <areaCount> il numero totale delle aree
    # - associare con <areaMap> il nome dell'area con un numero sequenziale rappresentante la colonna della matrice di output
    areaList = dao.getAreaListFromDb()
    for area in areaList:
        scanCount = scanCount + area['lastScanId']
        areaName = area['area']
        areaMap[areaName] = areaCount
        areaCount = areaCount + 1
    
    #step 3
    # - crea una matrice <inputMatrix> fatta di <scanCount> righe e di <wifiCount> colonne, con valori tutti a zero
    # - crea una matrice <outputMatrix> fatta di <scanCount> righe e di <areaCount> colonne, con valori tutti a zero
    inputMatrix = np.zeros((scanCount, wifiCount))
    outputMatrix = np.zeros((scanCount, areaCount))
    
    #step 4
    #ottiene le coppie aree-scansioni uniche, e le itera;
    #ad ogni iterazione:
    # - ottiene una rowIndex sequenziale
    # - ottiene la columnIndex dalla areaMap in base al nome dell'area
    # - assegna il valore areaId al vettore di uscita outputVector[rowIndex, columnIndex]
    # - esegue lo step 5
    rowIndex = 0
    areaScanList = dao.getAreaAndScanIdListFromDb()
    for areaScan in areaScanList:
        areaName = areaScan["area"]
        columnIndex = areaMap[areaName]
        outputMatrix[rowIndex, columnIndex] = 1.0
        
        #step 5
        #ottiene le scansioni per l'area e lo scanId correnti, e le itera;
        #ad ogni iterazione:
        # - ottiene la columnIndex dalla wifiMap in base al wifiName
        # - assegna il valore wifiLevel alla matrice di ingresso inputMatrix[rowIndex, columnIndex] 
        scanId   = areaScan["scanId"] 
        scanList = dao.getScansFromDb(areaName, scanId)
        for scan in scanList:
            wifiName = scan["wifiName"]
            columnIndex = wifiMap[wifiName]
            wifiLevel = scan["wifiLevel"] 
            inputMatrix[rowIndex, columnIndex] = wifiLevel

        rowIndex = rowIndex + 1
    
    #torna le matrici di input e output
    return inputMatrix, outputMatrix
    

#costruisce una rete neurale con keras
def buildAndFitAnn(inputMatrix, outputMatrix):
    
    global classifier
    
    #calcola: 
    #numero di neuroni di input in base al numero di colonne di inputMatrix
    #numero di neuroni di output in base al numero di colonne di outputMatrix
    #un numero di neuroni hidden proporzionale all'input ed all'output
    inputUnits = inputMatrix.shape[1]
    outputUnits = outputMatrix.shape[1]
    hidden1Units = math.ceil((inputUnits + outputUnits) / 3 * 2)
    hidden2Units = math.ceil((hidden1Units + outputUnits) /3)

    #Normalizza la matrice di ingresso
    inputMatrix = StandardScaler().fit_transform(inputMatrix)
    
    #Inizializza la rete neurale
    classifier = Sequential()
    
    #aggiunge lo strato di input ed il primo strato nascosto
    classifier.add(Dense(units = hidden1Units, kernel_initializer = 'uniform', activation = 'relu', input_dim = inputUnits))
    
    #aggiunge il secondo strato nascosto
    classifier.add(Dense(units = hidden2Units, kernel_initializer = 'uniform', activation = 'relu'))
    
    #aggiunge lo strato di uscita
    classifier.add(Dense(units = outputUnits, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    #compila la rete neurale
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #addestra la rete neurale
    classifier.fit(inputMatrix, outputMatrix, batch_size = 10, epochs = 100)
    

#crea la matrice di input in base alle scansioni passate in ingresso
def makeInputMatrixFromScans(wifiScans):
    
    #inizializza a zero la matrice di ingresso    
    inputMatrix = np.zeros((1, wifiCount))
    
    #effettua un ciclo sulle scansioni passate in ingresso
    for wifiScan in wifiScans:
        
        #ottiene i dettagli della singola scansione
        wifiName = wifiScan["wifiName"]
        wifiLevel = wifiScan["wifiLevel"]
        print("wifi name: ", wifiName, " level: ", wifiLevel)

        #se la wifiName e' tra quelle utilizzate per il training..        
        if wifiName in wifiMap:
            
            #ottiene l'indice della matrice di input corrispondente al wifiName
            columnIndex = wifiMap[wifiName]
        
            #popola l'elemento columnIndex dell'inputMatrix con il valore wifiLevel
            inputMatrix[0, columnIndex] = wifiLevel
    
    #torna la matrice di input
    return inputMatrix
    

#effettua una predizione
def predictArea(inputMatrix):
    
    global classifier
    
    #Normalizza la matrice di ingresso
    inputMatrix = StandardScaler().fit_transform(inputMatrix)
    
    #effettua la previsione
    outputPredictMatrix = classifier.predict(inputMatrix)
    print("previsione: ", outputPredictMatrix)
    
    #scorre le aree e sceglie quella con maggiore probabilita'
    predictArea = {}
    maxPredictProbability = 0
    predictIndex = 0
    areaList = dao.getAreaListFromDb()
    for area in areaList:

        if outputPredictMatrix[0, predictIndex] > maxPredictProbability:
            maxPredictProbability = outputPredictMatrix[0, predictIndex]
            predictArea = area

        predictIndex = predictIndex + 1

    #torna l'area predetta
    return predictArea

