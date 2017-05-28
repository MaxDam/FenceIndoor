
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
#import petl as etl
import dbEngine as dao


#rete neurale artificiale
classifier = None


#costruisce la matrice di input ed il vettore di output 
#in base ai dati presenti sul db
def makeDataFromDb():
    
    #step 1 
    #ottiene l'elenco delle wifi acquisite  nelle scansioni, e le itera per ottenere:
    # - <wifiCount> il numero di wifi acquisite
    # - una dictionary <wifiMap> che associa ogni <wifiName> con un numero sequenziale rappresentante la colonna della matrice di input
    wifiCount = 0
    wifiMap = {}
    wifiList = dao.getWifiListFromDb()
    for wifi in wifiList:
        wifiCount = wifiCount + 1
        wifiName = wifi['wifiName']
        wifiMap[wifiName] = wifiCount
    
    #step 2
    #ottiene le aree dal database, e le itera, per ottenere:
    # - <scanCount> la sommatoria dei <lastScanId> delle aree
    # - <areaCount> il numero totale delle aree
    # - <areaMap> una mappa che associa il nome dell'area ad un indice progressivo
    areaCount = 0
    scanCount = 0
    areaMap = {}
    areaList = dao.getAreaListFromDb()
    for area in areaList:
        areaCount = areaCount + 1
        scanCount = scanCount + area['lastScanId']
        areaName = area['area']
        areaMap[areaName] = areaCount
    
    #step 3
    # - crea una matrice <inputMatrix> fatta di <scanCount> righe e di <wifiCount> colonne, con valori tutti a zero
    # - crea una matrice <outputMatrix> fatta di <scanCount> righe e di <areaCount> colonne, con valori tutti a zero
    inputMatrix = np.zeros(wifiCount, scanCount, dtype=np.int)
    outputMatrix = np.zeros(areaCount, scanCount, dtype=np.int)
    
    #step 4
    #ottiene le coppie aree-scansioni uniche, e le itera
    #ad ogni iterazione:
    # - ottiene una rowIndex sequenziale
    # - ottiene la columnIndex dalla areaMap in base al nome dell'area
    # - assegna il valore areaId al vettore di uscita outputVector[rowIndex, columnIndex]
    rowIndex = -1
    areaScanList = dao.getAreaAndScanIdListFromDb()
    for areaScan in areaScanList:
        rowIndex = rowIndex + 1
        areaName = areaScan["area"]
        columnIndex = areaMap[areaName]
        outputMatrix[rowIndex, columnIndex]
        
        #step 5
        #ad ogni iterazione..
        #ottiene le scansioni per l'area e lo scanId correnti, e le itera
        #ad ogni iterazione:
        # - ottiene la columnIndex dalla wifiMap in base al nome della wifi
        # - assegna il valore wifiLevel alla matrice di ingresso inputMatrix[rowIndex, columnIndex] 
        scanId   = areaScan["scanId"] 
        scanList = dao.getScansFromDb(areaName, scanId)
        for scan in scanList:
            wifiName = scan["wifiName"]
            columnIndex = wifiMap[wifiName]
            wifiLevel = scan["wifiLevel"] 
            inputMatrix[rowIndex, columnIndex] = wifiLevel
    
    return inputMatrix, outputMatrix
    

#costruisce una rete neurale con keras
def buildAndFitAnn(inputMatrix, outputMatrix):
    
    #TODO 
    #numero di neuroni di input in base al numero di colonne di inputMatrix
    #numero di neuroni di output in base al numero di colonne di outputMatrix
    #un numero di neuroni hidden proporzionale all'input ed all'output

    #TODO ottenere outputVector da outputMatrix.. classificazione OneClass
    outputVector = np.zeros(1000, dtype=np.int)
    
    global classifier
    
    #Normalizza la matrice di ingresso
    inputMatrix = StandardScaler().fit_transform(inputMatrix)
    
    #Inizializza la rete neurale
    classifier = Sequential()
    
    #aggiunge lo strato di input ed il primo strato nascosto
    classifier.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
    
    #aggiunge il secondo strato nascosto
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    
    #aggiunge lo strato di uscita
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    
    #compila la rete neurale
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    #addestra la rete neurale
    classifier.fit(inputMatrix, outputVector, batch_size = 10, epochs = 100)
    

#crea la matrice di input in base alle scansioni passate in ingresso
def makeInputMatrixFromScans(wifiScans):
    
    #TODO da implementare
    
    inputMatrix = np.zeros(15,100, dtype=np.int)
    return inputMatrix
    

#effettua una predizione
def predictArea(inputMatrix):
    
    global classifier
    
    y_pred = classifier.predict(inputMatrix)
    y_pred = (y_pred > 0.5)
    
    #TODO da implementare
    
    #prepara la risposta
    area = {}
    area['name'] = "Non lo so ancora.. abbi un poco di pazienza";
    
    #torna l'area predetta
    return area

