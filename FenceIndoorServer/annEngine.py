
#funzioni di utilita' ANN e di preparazione dati ETL

import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 
#import petl as etl
import dbEngine as dao
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.models import load_model
#from keras.optimizers import SGD
from keras.optimizers import Adam
#from keras.utils import np_utils

#classificatore rete neurale artificiale
classifier = None

#standard scale
scaler = None

#una dictionary ogni <wifiName> con un numero sequenziale rappresentante la colonna della matrice di input
wifiMapEncode = {}

#una dictionary che data la posizione dell'area ritorna i dettagli dell'area
areaMapDecode = {}
  
#file che conterra' in dictionary di wifiMapEncode
wifiMapFile = 'tmp/wifiMap.json' 

#file che conterra' in dictionary di areaMapDecode
areaMapFile = 'tmp/areaMap.json' 

#file pickle che conterrà lo scaler salvato
scalerFile = 'tmp/scaler.pkl'

#file json che conterra' la struttura della rete
classifierFile='tmp/classifier.json'

#file h5 che conterra' i pesi di addestramento salvati
classifierWeightFile='tmp/classifier.h5'

#costruisce la matrice di input ed il vettore di output 
#in base ai dati presenti sul db
def makeDataFromDb():
    
    global wifiMapEncode, areaMapDecode
    
    #step 1 
    #ottiene l'elenco delle wifi acquisite  nelle scansioni, e le itera per:
    # - assegnare a <wifiCount> il numero totale delle wifi acquisite
    # - associare con <wifiMap> ogni <wifiName> con un numero sequenziale rappresentante la colonna della matrice di input
    wifiMapEncode = {}
    wifiCount = 0 
    wifiList = dao.getWifiListFromDb()
    for wifi in wifiList:
        wifiName = wifi['wifiName']
        wifiMapEncode[wifiName] = wifiCount
        wifiCount = wifiCount + 1
        
    #step 2
    #ottiene le aree dal database, e le itera per:
    # - assegnare a <scanCount> la sommatoria dei <lastScanId> delle aree
    # - assegnare ad <areaCount> il numero totale delle aree
    # - associare con <areaMapEncode> il nome dell'area con un numero sequenziale rappresentante la colonna della matrice di output
    # - associare con <areaMapDecode> il numero sequenziale rappresentante la colonna della matrice di output con i dettagli dell'area
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
        for scan in scanList:
            wifiName = scan["wifiName"]
            columnIndex = wifiMapEncode[wifiName]
            wifiLevel = scan["wifiLevel"] 
            inputMatrix[rowIndex, columnIndex] = wifiLevel

        rowIndex = rowIndex + 1
    
    #torna le matrici di input e output
    return inputMatrix, outputMatrix
    

#costruisce una rete neurale con keras
def buildAndFitAnn(inputMatrix, outputMatrix):
    
    global classifier, scaler
    
    #calcola: 
    #numero di neuroni di input in base al numero di colonne di inputMatrix
    #numero di neuroni di output in base al numero di colonne di outputMatrix
    #un numero di neuroni hidden proporzionale all'input ed all'output
    inputUnits = inputMatrix.shape[1]
    outputUnits = outputMatrix.shape[1]
    hidden1Units = int((inputUnits + outputUnits) / 2 * 3)
    hidden2Units = int((inputUnits + outputUnits) / 2)
    hidden34Units = int((hidden2Units + outputUnits) / 2)

    #log
    print("matrice di input:")
    print(inputMatrix)
    
    #Normalizza la matrice di ingresso
    inputMatrix = inputMatrix.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    inputMatrix = scaler.fit_transform(inputMatrix)
    
    #converte la classe vettore in classe matrice binaria
    #outputMatrix = np_utils.to_categorical(outputMatrix, outputUnits)
    
    #log
    print("matrice di input normalizzata:")
    print(inputMatrix)
    
    #log
    print("matrice di output:")
    print(outputMatrix)
    
    #Inizializza la rete neurale
    classifier = Sequential()
    
    #aggiunge lo strato di input ed il primo strato nascosto    
    classifier.add(Dense(hidden1Units, input_shape=(inputUnits,)))
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.3))
    
    #aggiunge il secondo strato nascosto
    classifier.add(Dense(hidden2Units))
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.3))
    
    #aggiunge il terzo strato nascosto
    classifier.add(Dense(hidden34Units))
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.3))
    
    #aggiunge il quarto strato nascosto
    classifier.add(Dense(hidden34Units))
    classifier.add(Activation('relu'))
    classifier.add(Dropout(0.3))

    #aggiunge lo strato di uscita
    classifier.add(Dense(outputUnits))
    classifier.add(Activation('softmax'))
    classifier.summary()

    #compila la rete neurale
    #classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    #classifier.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
    classifier.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    #addestra la rete neurale
    #classifier.fit(inputMatrix, outputMatrix, batch_size=128, epochs=650, verbose=1)
    classifier.fit(inputMatrix, outputMatrix, batch_size=128, epochs=200, verbose=1)

    #salva la rete neurale su files
    saveAnnToFiles()
        
#crea la matrice di input in base alle scansioni passate in ingresso
def makeInputMatrixFromScans(wifiScans):
    
    global wifiMapEncode
    
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
    
    #torna la matrice di input
    return inputMatrix
    

#effettua una predizione
def predictArea(inputMatrix):
    
    global classifier, scaler, areaMapDecode
    
    #log
    print("matrice di input:")
    print(inputMatrix)
    
    #Normalizza la matrice di ingresso
    inputMatrix = scaler.transform(inputMatrix)
    
    #log
    print("matrice di input normalizzata:")
    print(inputMatrix)
    
    #effettua la previsione
    outputPredictMatrix = classifier.predict(inputMatrix)
    
    #log
    print("previsione (",outputPredictMatrix.shape[1], " risultati ):")
    #print(', '.join('{:0.2f}%'.format(i*100) for i in outputPredictMatrix[0]))
    
    #scorre i risultati per ottenere quello con maggione probabilita'
    maxPredictProbability = 0
    predictArea = {}
    for predictIndex in range(0, outputPredictMatrix.shape[1]):
        
        #stampa l'area con la probabilita' associata
        print(areaMapDecode[str(predictIndex)]["area"], ": ", '{:0.2f}%'.format(outputPredictMatrix[0, predictIndex]*100));
        
        #se l'area ha una probabilita' maggiore delle precedenti è quella scelta
        if outputPredictMatrix[0, predictIndex] > maxPredictProbability:
            maxPredictProbability = outputPredictMatrix[0, predictIndex]
            predictArea = areaMapDecode[str(predictIndex)]
    
    #torna l'area con maggiore probabilita'
    return predictArea


#salva la rete neurale in alcuni files
def saveAnnToFiles():
    
    global wifiMapEncode, areaMapDecode, classifier, scaler

    #salva il file con i mapping delle reti con le colonne della matrice
    json.dump(wifiMapEncode, open(wifiMapFile,'w'))
    
    #salva il file con i mapping che data la colonna della matrice ritorna l'area
    json.dump(areaMapDecode, open(areaMapFile,'w'))

    #salva lo scaler in un file pickle
    joblib.dump(scaler, scalerFile)
    
    #salva la struttura della rete in json
    classifier.save(classifierFile)
    
    #salva i pesi in un file h5
    classifier.save_weights(classifierWeightFile)


#carica la rete neurale dai files salvati
def loadAnnFromFiles():
    
    global wifiMapEncode, areaMapDecode, classifier, scaler
    
    #ricostruisce wifiMap
    wifiMapEncode = json.load(open(wifiMapFile))
    
    #ricostruisce areaMap
    areaMapDecode = json.load(open(areaMapFile))

    #ricostruisce lo scaler
    scaler = joblib.load(scalerFile) 
    
    #ricostruisce la struttura della rete neurale
    classifier = load_model(classifierFile)

    #ricostruisce i pesi della ann
    classifier.load_weights(classifierWeightFile)


#elegge l'area predetta da una lista di aree predette
def electPredictArea(predictAreaList):
    
    print("Final predict area list:")
    print(predictAreaList)
    
    #cerca l'area con maggiori occorrenze trovate nella lista e la elegge come area predetta
    maxNumVotes = 0
    electArea = {}
    for area in predictAreaList:
        numVotes = predictAreaList.count(area)
        if(numVotes > maxNumVotes):
            maxNumVotes = numVotes
            electArea = area
            
    return electArea
        