
#funzioni di utilita' ANN e di preparazione dati ETL

import json
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib 
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
import dataLayer as dl

#modello rete neurale artificiale
model = None

#scaler
scaler = None

#una dictionary ogni <wifiName> con un numero sequenziale rappresentante la colonna della matrice di input
wifiMapEncode = {}

#una dictionary che data la posizione dell'area ritorna i dettagli dell'area
areaMapDecode = {}
  
#file che conterranno le dictionary
wifiMapFile = 'tmp/wifiMap.json' 
areaMapFile = 'tmp/areaMap.json' 

#file che conterranno i parametri della rete neurale
scalerFile      = 'tmp/scaler.pkl'
modelFile       = 'tmp/model.json'
modelWeightFile = 'tmp/model.h5'

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
    wifiList = dl.getWifiListFromDb()
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
    areaList = dl.getAreaListFromDb()
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
    dl.indexScans()
    print("acquisizione delle scansioni..")
    rowIndex = 0
    columnIndex = 0
    areaScanList = dl.getAreaAndScanIdListFromDb()
    for areaScan in areaScanList:
        areaName = areaScan["area"]
        columnIndex = areaMapEncode[areaName]
        outputMatrix[rowIndex, columnIndex] = 1
        
        #step 5
        #ottiene le scansioni per l'area e lo scanId correnti, e le itera;
        #ad ogni iterazione:
        # - ottiene la columnIndex dalla wifiMap in base al wifiName
        # - assegna il valore wifiLevel alla matrice di ingresso inputMatrix[rowIndex, columnIndex] 
        scanId   = areaScan["scanId"] 
        scanList = dl.getScansFromDb(areaName, scanId)
        #print("(", rowIndex, " di ",len(areaScanList),") ottenute", len(scanList), " scansioni per l'area", areaName, " con scanId", scanId)
        for scan in scanList:
            wifiName = scan["wifiName"]
            columnIndex = wifiMapEncode[wifiName]
            wifiLevel = scan["wifiLevel"] 
            inputMatrix[rowIndex, columnIndex] = wifiLevel

        #incrementa il numero di riga
        rowIndex = rowIndex + 1
    print("preparate le matrici con ", len(areaScanList), " scansioni")
    
    #log
    print("FINE PREPARAZIONE DATI")
    
    #torna le matrici di input e output
    return inputMatrix, outputMatrix
       
#costruisce una rete neurale con keras
def buildAndFitAnn(inputMatrix, outputMatrix):
    
    global model, scaler
    
    #log
    print("INIZIO ADDESTRAMENTO ANN")
    
    #hyperparameters
    numberHiddenLayers = 3 #10
    dropout = 0.5 #0.3
    batch_size = 32 #128
    epochs = 50
    test_split = 0.33
    validation_split = 0.33
    
    #log
    #print("matrice di input:")
    #print(inputMatrix)
    
    #Normalizza la matrice di ingresso
    inputMatrix = inputMatrix.astype('float32')
    #scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(0, 1))
    inputMatrix = scaler.fit_transform(inputMatrix)
    
    #calcola: 
    #numero di neuroni di input in base al numero di colonne di inputMatrix
    #numero di neuroni di output in base al numero di colonne di outputMatrix
    #un numero di neuroni hidden proporzionale all'input ed all'output
    inputUnits = inputMatrix.shape[1]
    outputUnits = outputMatrix.shape[1]
    hiddenUnits = int(inputUnits * 1.5)
    
    #effettua lo split dei dati di train con quelli di test
    inputMatrix, inputTestMatrix, outputMatrix, outputTestMatrix = train_test_split(inputMatrix, outputMatrix, test_size=test_split, random_state=42)
    
    #log
    #print("matrice di input normalizzata:")
    #print(inputMatrix)
    
    #log
    #print("matrice di output:")
    #print(outputMatrix)
    
    #aggiunge lo strato di input ed il primo strato nascosto + una regolarizzazione l2   
    input = Input(shape=(inputUnits,))
    first = Dense(hiddenUnits)(input)
    first = BatchNormalization()(first)
    first = Activation('tanh')(first)
    first = Dropout(dropout)(first)
    
    #aggiunge numberHiddenLayer strati nascosti
    hidden = first
    for _ in range(numberHiddenLayers):
        #aggiunge lo strato nascosto
        hidden = Dense(hiddenUnits)(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Activation('tanh')(hidden)
        hidden = Dropout(dropout)(hidden)
        
    #aggiunge lo strato di uscita
    output = hidden
    output = Dense(outputUnits)(output)
    output = BatchNormalization()(output)
    output = Activation('softmax')(output)
    
    #crea il modello
    model = Model(inputs=input, outputs=output)

    #compila la rete neurale
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    #addestra la rete neurale
    model.fit(inputMatrix, outputMatrix, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=1)

    #stampa il risultato della valutazione del modello
    score = model.evaluate(inputTestMatrix, outputTestMatrix, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    printScores(inputTestMatrix, outputTestMatrix)

    #salva la rete neurale su files
    saveAnnToFiles()
    
    #log
    print("FINE ADDESTRAMENTO ANN")
        
#crea la matrice di input in base alle scansioni passate in ingresso
def makeInputMatrixFromScans(wifiScans):
    
    global wifiMapEncode
    
    #log
    #print("INIZIO PREPARAZIONE DATI")
    
    #ricostruisce la rete dai files, se necessario
    if model is None:
        loadAnnFromFiles()
    
    #inizializza a zero la matrice di ingresso 
    inputMatrix = np.zeros((1, len(wifiMapEncode)))    
    
    #effettua un ciclo sulle scansioni passate in ingresso
    for wifiScan in wifiScans:
        
        #ottiene i dettagli della singola scansione
        wifiName = wifiScan["wifiName"]
        wifiLevel = wifiScan["wifiLevel"]
        #print("wifi name: ", wifiName, " level: ", wifiLevel)

        #se la wifiName e' tra quelle utilizzate per il training..        
        if wifiName in wifiMapEncode:
            
            #ottiene l'indice della matrice di input corrispondente al wifiName
            columnIndex = wifiMapEncode[wifiName]
        
            #popola l'elemento columnIndex dell'inputMatrix con il valore wifiLevel
            inputMatrix[0, columnIndex] = wifiLevel
    
    #log
    #print("FINE PREPARAZIONE DATI")
    
    #torna la matrice di input
    return inputMatrix
    

#effettua una predizione
def predictArea(inputMatrix):
    
    global model, scaler, areaMapDecode
    
    #log
    #print("INIZIO PREDIZIONE ANN")
    
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
    outputPredictMatrix = model.predict(inputMatrix)
    
    #log
    print("matrice di previsione:")
    print(outputPredictMatrix)
    
    #ottiene un vettore con somma di tutti gli output (axis=0)
    sumOutputPredictVector = np.sum(outputPredictMatrix, axis=0)
    print("vettore con somma di tutti gli output:")
    print(sumOutputPredictVector)

    #ottiene l'indice della classe con la massima previsione
    maxPredictionIndex = np.argmax(sumOutputPredictVector)
    print("indice della classe con la massima previsione:")
    print(maxPredictionIndex)
    print()

    #ottiene l'area a massima previsione dato l'indice
    predictArea = areaMapDecode[str(maxPredictionIndex)]
     
    #log
    #print("FINE PREDIZIONE ANN")
    
    #torna l'area con maggiore probabilita'
    return predictArea


#salva la rete neurale in alcuni files
def saveAnnToFiles():
    
    global wifiMapEncode, areaMapDecode, model, scaler

    #salva il file con i mapping delle reti con le colonne della matrice
    json.dump(wifiMapEncode, open(wifiMapFile,'w'))
    
    #salva il file con i mapping che data la colonna della matrice ritorna l'area
    json.dump(areaMapDecode, open(areaMapFile,'w'))

    #salva lo scaler in un file pickle
    joblib.dump(scaler, scalerFile)
    
    #salva la struttura della rete in json
    model.save(modelFile)
    
    #salva i pesi della rete in un file h5
    model.save_weights(modelWeightFile)


#carica la rete neurale dai files salvati
def loadAnnFromFiles():
    
    global wifiMapEncode, areaMapDecode, model, scaler
    
    #ricostruisce wifiMap
    wifiMapEncode = json.load(open(wifiMapFile))
    
    #ricostruisce areaMap
    areaMapDecode = json.load(open(areaMapFile))

    #ricostruisce lo scaler
    scaler = joblib.load(scalerFile) 
    
    #ricostruisce la struttura della rete neurale
    model = load_model(modelFile)
    
    #ricostruisce i pesi della ann
    model.load_weights(modelWeightFile)
  

def printScores(X_test, Y_test):
    
    global model

    #prepara i dati
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    
    #calcola gli score
    accuracy_score_val = accuracy_score(Y_test, Y_pred)
    precision_score_val = precision_score(Y_test, Y_pred, average='weighted') # tp / (tp + fp)
    recall_score_val = recall_score(Y_test, Y_pred, average='weighted') # tp / (tp + fn)
    f1_score_val = f1_score(Y_test, Y_pred, average='weighted')

    #stampa gli score
    print("accuracy_score: %0.4f" % accuracy_score_val)
    print("precision_score: %0.4f" % precision_score_val)
    print("recall_score: %0.4f" % recall_score_val)
    print("f1_score: %0.4f" % f1_score_val)
    print("confusion matrix:")
    print(confusion_matrix(Y_test, Y_pred))
    