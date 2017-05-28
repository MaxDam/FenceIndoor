
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
#import petl as etl
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
        wifiCount = wifiCount + 1
        wifiName = wifi['wifiName']
        wifiMap[wifiName] = wifiCount
    
    #step 2
    #ottiene le aree dal database, e le itera per:
    # - assegnare a <scanCount> la sommatoria dei <lastScanId> delle aree
    # - assegnare ad <areaCount> il numero totale delle aree
    # - associare con <areaMap> il nome dell'area con un numero sequenziale rappresentante la colonna della matrice di output
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
    #ottiene le coppie aree-scansioni uniche, e le itera;
    #ad ogni iterazione:
    # - ottiene una rowIndex sequenziale
    # - ottiene la columnIndex dalla areaMap in base al nome dell'area
    # - assegna il valore areaId al vettore di uscita outputVector[rowIndex, columnIndex]
    # - esegue lo step 5
    rowIndex = -1
    areaScanList = dao.getAreaAndScanIdListFromDb()
    for areaScan in areaScanList:
        rowIndex = rowIndex + 1
        areaName = areaScan["area"]
        columnIndex = areaMap[areaName]
        outputMatrix[rowIndex, columnIndex]
        
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
    hidden1Units = int((inputUnits + outputUnits) / 2)
    hidden2Units = int((hidden1Units + outputUnits) /2)

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
    inputMatrix = np.zeros(wifiCount, dtype=np.int)
    
    #effettua un ciclo sulle scansioni passate in ingresso
    for wifiScan in wifiScans:
        
        #ottiene i dettagli della singola scansione
        wifiName = wifiScan["wifiName"]
        wifiLevel = wifiScan["wifiLevel"]
        print("wifi name: ", wifiName, " level: ", wifiLevel)
        
        #ottiene l'indice della matrice di input corrispondente al wifiName
        columnIndex = wifiMap[wifiName]
        
        #popola l'elemento columnIndex dell'inputMatrix con il valore wifiLevel
        inputMatrix[columnIndex] = wifiLevel
    
    #torna la matrice di input
    return inputMatrix
    

#effettua una predizione
def predictArea(inputMatrix):
    
    global classifier
    
    #Normalizza la matrice di ingresso
    inputMatrix = StandardScaler().fit_transform(inputMatrix)
    
    #effettua la previsione
    y_pred = classifier.predict(inputMatrix)
    y_pred = (y_pred > 0.5)
    
    #TODO da implementare

    #prepara la risposta
    area = {}
    area['name'] = "Non lo so ancora.. abbi un poco di pazienza";
    
    #torna l'area predetta
    return area

