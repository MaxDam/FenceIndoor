
#funzioni di utilita' ANN e di preparazione dati ETL

import json
import numpy as np
#import petl as etl
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import Normalizer
from sklearn.externals import joblib 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.models import load_model
from keras.optimizers import Adam
from keras import regularizers
import dbEngine as dao

#classificatore rete neurale artificiale
classifier = None

#scaler
scaler = None

#normalizer
normalizer = None

#una dictionary ogni <wifiName> con un numero sequenziale rappresentante la colonna della matrice di input
wifiMapEncode = {}

#una dictionary che data la posizione dell'area ritorna i dettagli dell'area
areaMapDecode = {}
  
#file che conterranno le dictionary
wifiMapFile = 'tmp/wifiMap.json' 
areaMapFile = 'tmp/areaMap.json' 

#file che conterranno i parametri della rete neurale
scalerFile          = 'tmp/scaler.pkl'
normalizerFile      = 'tmp/normalizer.pkl'
classifierModelFile = 'tmp/classifier.json'
classifierWeightFile= 'tmp/classifier.h5'

#set del seed per la randomizzazione
np.random.seed(1671)

#costruisce la matrice di input ed il vettore di output 
#in base ai dati presenti sul db
def makeDataFromDb():
    
    global wifiMapEncode, areaMapDecode
    
    #log
    print("INIZIO PREPARAZIONE DATI")
    
    #step 1 
    #ottiene l'elenco delle wifi acquisite  nelle scansioni, e le itera per:
    # - assegnare a <wifiCount> il numero totale delle wifi acquisite
    # - associare con <wifiMap> ogni <wifiName> con un numero sequenziale rappresentante la colonna della matrice di input
    print("acquisizione delle wifi scansionate..")
    wifiMapEncode = {}
    wifiCount = 0 
    wifiList = dao.getWifiListFromDb()
    for wifi in wifiList:
        wifiName = wifi['wifiName']
        wifiMapEncode[wifiName] = wifiCount
        wifiCount = wifiCount + 1
    print("scansionate ", wifiCount, " wifi")
        
    #step 2
    #ottiene le aree dal database, e le itera per:
    # - assegnare a <scanCount> la sommatoria dei <lastScanId> delle aree
    # - assegnare ad <areaCount> il numero totale delle aree
    # - associare con <areaMapEncode> il nome dell'area con un numero sequenziale rappresentante la colonna della matrice di output
    # - associare con <areaMapDecode> il numero sequenziale rappresentante la colonna della matrice di output con i dettagli dell'area
    print("acquisizione delle aree..")
    areaMapEncode = {}
    areaMapDecode = {}
    scanCount = 0
    areaCount = 0
    areaList = dao.getAreaListFromDb()
    for area in areaList:
        scanCount = scanCount + area['lastScanId']
        areaName = area['area']
        if areaName == '': continue
        areaMapEncode[areaName] = areaCount
        areaMapDecode[str(areaCount)] = area
        areaCount = areaCount + 1
    print("predisposte ", areaCount, " aree")
    
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
    print("indicizzazione delle scansioni..")
    dao.indexScans()
    print("acquisizione delle scansioni..")
    rowIndex = 0
    columnIndex = 0
    areaScanList = dao.getAreaAndScanIdListFromDb()
    for areaScan in areaScanList:
        areaName = areaScan["area"]
        columnIndex = areaMapEncode[areaName]
        outputMatrix[rowIndex, columnIndex] = 1.0
        
        #step 5
        #ottiene le scansioni per l'area e lo scanId correnti, e le itera;
        #ad ogni iterazione:
        # - ottiene la columnIndex dalla wifiMap in base al wifiName
        # - assegna il valore wifiLevel alla matrice di ingresso inputMatrix[rowIndex, columnIndex] 
        scanId   = areaScan["scanId"] 
        scanList = dao.getScansFromDb(areaName, scanId)
        print("(", rowIndex, " di ",len(areaScanList),") ottenute", len(scanList), " scansioni per l'area", areaName, " con scanId", scanId)
        for scan in scanList:
            wifiName = scan["wifiName"]
            columnIndex = wifiMapEncode[wifiName]
            wifiLevel = scan["wifiLevel"] 
            inputMatrix[rowIndex, columnIndex] = wifiLevel

        rowIndex = rowIndex + 1
    print("preparate le matrici con ", len(areaScanList), " scansioni")
    
    #log
    print("FINE PREPARAZIONE DATI")
    
    #torna le matrici di input e output
    return inputMatrix, outputMatrix
    

#costruisce una rete neurale con keras
def buildAndFitAnn(inputMatrix, outputMatrix):
    
    global classifier, scaler, normalizer
    
    #log
    print("INIZIO PREPARAZIONE/ADDESTRAMENTO ANN")
    
    #calcola: 
    #numero di neuroni di input in base al numero di colonne di inputMatrix
    #numero di neuroni di output in base al numero di colonne di outputMatrix
    #un numero di neuroni hidden proporzionale all'input ed all'output
    inputUnits = inputMatrix.shape[1]
    outputUnits = outputMatrix.shape[1]
    #hiddenUnits = int((inputUnits + outputUnits) / 2)
    hiddenUnits = int(inputUnits + outputUnits)
    
    #numero di strati nascosti della rete
    numberHiddenLayers = 10
    
    #log
    print("matrice di input:")
    print(inputMatrix)
    
    #Normalizza la matrice di ingresso
    inputMatrix = inputMatrix.astype('float32')
    scaler = StandardScaler()
    inputMatrix = scaler.fit_transform(inputMatrix)
    
    #log
    print("matrice di input normalizzata:")
    print(inputMatrix)
    
    #log
    print("matrice di output:")
    print(outputMatrix)
    
    #Inizializza la rete neurale
    classifier = Sequential()
    
    #aggiunge lo strato di input ed il primo strato nascosto + una regolarizzazione l2   
    #classifier.add(Dense(hiddenUnits, input_shape=(inputUnits,)))
    classifier.add(Dense(hiddenUnits, input_dim=inputUnits, kernel_regularizer=regularizers.l2(0.01)))
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.3))
    
    #aggiunge numberHiddenLayer strati nascosti
    for i in range(numberHiddenLayers):
        #aggiunge lo strato nascosto
        classifier.add(Dense(hiddenUnits))
        classifier.add(Activation('relu'))
        classifier.add(Dropout(0.3))

    #aggiunge lo strato di uscita
    classifier.add(Dense(outputUnits))
    classifier.add(Activation('softmax'))
    classifier.summary()

    #compila la rete neurale
    classifier.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    #addestra la rete neurale
    classifier.fit(inputMatrix, outputMatrix, batch_size=128, epochs=200, verbose=1)

    #salva la rete neurale su files
    saveAnnToFiles()
    
    #log
    print("FINE PREPARAZIONE/ADDESTRAMENTO ANN")
        
#crea la matrice di input in base alle scansioni passate in ingresso
def makeInputMatrixFromScans(wifiScans):
    
    global wifiMapEncode
    
    #log
    print("INIZIO PREPARAZIONE DATI")
    
    #ricostruisce la rete dai files, se necessario
    if classifier is None:
        loadAnnFromFiles()
    
    #inizializza a zero la matrice di ingresso 
    inputMatrix = np.zeros((1, len(wifiMapEncode)))    
    
    #effettua un ciclo sulle scansioni passate in ingresso
    for wifiScan in wifiScans:
        
        #ottiene i dettagli della singola scansione
        wifiName = wifiScan["wifiName"]
        wifiLevel = wifiScan["wifiLevel"]
        print("wifi name: ", wifiName, " level: ", wifiLevel)

        #se la wifiName e' tra quelle utilizzate per il training..        
        if wifiName in wifiMapEncode:
            
            #ottiene l'indice della matrice di input corrispondente al wifiName
            columnIndex = wifiMapEncode[wifiName]
        
            #popola l'elemento columnIndex dell'inputMatrix con il valore wifiLevel
            inputMatrix[0, columnIndex] = wifiLevel
    
    #log
    print("FINE PREPARAZIONE DATI")
    
    #torna la matrice di input
    return inputMatrix
    

#effettua una predizione
def predictArea(inputMatrix):
    
    global classifier, scaler, normalizer, areaMapDecode
    
    #log
    print("INIZIO PREDIZIONE ANN")
    
    #log
    print("matrice di input:")
    print(inputMatrix)
    
    #Normalizza la matrice di ingresso
    inputMatrix = scaler.transform(inputMatrix)
    
    #predispone la matrice di uscita
    outputPredictMatrix = np.zeros((1, len(areaMapDecode)))
    
    #log
    #print("matrice di input normalizzata:")
    #print(inputMatrix)
    
    #effettua la previsione
    outputPredictMatrix = classifier.predict(inputMatrix)
    
    #log
    #print("matrice di previsione:")
    #print(outputPredictMatrix)
    
    #ottiene la matrice di previsione massima
    maxPredictionMatrix = np.argmax(outputPredictMatrix, axis=1)
    print("matrice indici di previsione massima:")
    print(maxPredictionMatrix)
    
    #ottiene l'indice di previsione massima considerando le occorrenze trovate
    maxPredictionIndex = np.argmax(np.bincount(maxPredictionMatrix))
    print("indice previsione massima:")
    print(maxPredictionIndex)
    
    #ottiene l'area a massima previsione dato l'indice
    predictArea = areaMapDecode[str(maxPredictionIndex)]
     
    #log
    print("FINE PREDIZIONE ANN")
    
    #torna l'area con maggiore probabilita'
    return predictArea


#salva la rete neurale in alcuni files
def saveAnnToFiles():
    
    global wifiMapEncode, areaMapDecode, classifier, scaler, normalizer

    #salva il file con i mapping delle reti con le colonne della matrice
    json.dump(wifiMapEncode, open(wifiMapFile,'w'))
    
    #salva il file con i mapping che data la colonna della matrice ritorna l'area
    json.dump(areaMapDecode, open(areaMapFile,'w'))

    #salva lo scaler in un file pickle
    joblib.dump(scaler, scalerFile)
    
    #salva il normalizer in un file pickle
    joblib.dump(normalizer, normalizerFile)
    
    #salva la struttura della rete in json
    classifier.save(classifierModelFile)
    
    #salva i pesi in un file h5
    classifier.save_weights(classifierWeightFile)


#carica la rete neurale dai files salvati
def loadAnnFromFiles():
    
    global wifiMapEncode, areaMapDecode, classifier, scaler, normalizer
    
    #ricostruisce wifiMap
    wifiMapEncode = json.load(open(wifiMapFile))
    
    #ricostruisce areaMap
    areaMapDecode = json.load(open(areaMapFile))

    #ricostruisce lo scaler
    scaler = joblib.load(scalerFile) 
    
    #ricostruisce il normalizer
    normalizer = joblib.load(normalizerFile) 
    
    #ricostruisce la struttura della rete neurale
    classifier = load_model(classifierModelFile)

    #ricostruisce i pesi della ann
    classifier.load_weights(classifierWeightFile)
  