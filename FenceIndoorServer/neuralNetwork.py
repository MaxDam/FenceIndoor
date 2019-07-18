
#rete neurale artificiale (ANN)

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
import tensorflow as tf

#modello rete neurale artificiale e graph di tensorflow
model = None
graph = None

#scaler
scaler = None

#una dictionary ogni <wifiName> con un numero sequenziale rappresentante la colonna della matrice di input
wifiMapEncode = None

#una dictionary che data la posizione dell'area ritorna i dettagli dell'area
areaMapDecode = None

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

    #ottiene dal db le wifi e le scorre..
    wifiList = dl.getWifiListFromDb()
    for wifi in wifiList:

        #ottiene la wifiName, se è vuota la salta
        wifiName = wifi['wifiName']
        if wifiName == '': continue

        #incrementa il numero wifi e salva le dictionary di mapping
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

    #ottiene dal db le aree e le scorre..
    areaList = dl.getAreaListFromDb()
    for area in areaList:

        #ottiene l'area, se è vuota la salta
        areaName = area['area']
        if areaName == '': continue

        #incrementa il numero di scansioni e salva le dictionary di mapping
        scanCount = scanCount + area['lastScanId']
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
    
    #indicizza le scansioni
    print("indicizzazione delle scansioni..")
    dl.indexScans()
    
    print("acquisizione delle scansioni..")
    rowIndex = 0
    columnIndex = 0
    
    #ottiene dal db e scorre le coppie area e scanId ..
    areaScanList = dl.getAreaAndScanIdListFromDb()
    for areaScan in areaScanList:
        
        #ottiene l'area, se non e' tra quelle presenti nel areaMapEncode, la salta
        areaName = areaScan["area"]
        if areaName not in areaMapEncode: continue
        
        #ottiene l'indice della colonna corrispondente all'area ed imposta il singolo valore ad 1 (one hot encode)
        columnIndex = areaMapEncode[areaName]
        outputMatrix[rowIndex, columnIndex] = 1
        
        #step 5
        #ottiene le scansioni per l'area e lo scanId correnti, e le itera;
        #ad ogni iterazione:
        # - ottiene la columnIndex dalla wifiMap in base al wifiName
        # - assegna il valore wifiLevel alla matrice di ingresso inputMatrix[rowIndex, columnIndex] 
        
        scanId  = areaScan["scanId"] 
        
        #ottiene dal db e scorre le scansioni..
        scanList = dl.getScansFromDb(areaName, scanId)
        #print("(", rowIndex, " di ",len(areaScanList),") ottenute", len(scanList), " scansioni per l'area", areaName, " con scanId", scanId)
        for scan in scanList:
            
            #ottiene la wifiName, se non e' tra quelle presenti nel wifiMapEncode, la salta
            wifiName = scan["wifiName"]
            if wifiName not in wifiMapEncode: continue
            
            #ottiene l'indice della colonna corrispondente alla wifi ed imposta il valore di segnale
            columnIndex = wifiMapEncode[wifiName]
            wifiLevel = scan["wifiLevel"] 
            inputMatrix[rowIndex, columnIndex] = wifiLevel

        #incrementa il numero di riga
        rowIndex = rowIndex + 1

    print("preparate le matrici con ", len(areaScanList), " scansioni")
    
    #log
    print("FINE PREPARAZIONE DATI")
    
    #salva le dictionary di mapping
    json.dump(wifiMapEncode, open('tmp/wifi_map.json','w'))
    json.dump(areaMapDecode, open('tmp/area_map.json','w'))

    #torna le matrici di input e output
    return inputMatrix, outputMatrix

      
#costruisce una rete neurale con keras
def buildAndFitAnn(inputMatrix, outputMatrix):
    
    global model, graph, scaler

    #carica il graph di default se non c'e'
    if graph is None: 
        graph = tf.get_default_graph()

    #hyperparameters
    numberHiddenLayers = 3 #10
    dropout = 0.5 #0.3
    batch_size = 16
    epochs = 100
    test_split = 0.33
    validation_split = 0.2
    
    #log
    #print("matrice di input:")
    #print(inputMatrix)
    
    #calcola: 
    #numero di neuroni di input in base al numero di colonne di inputMatrix
    #numero di neuroni di output in base al numero di colonne di outputMatrix
    #un numero di neuroni hidden proporzionale all'input ed all'output
    inputUnits = inputMatrix.shape[1]
    outputUnits = outputMatrix.shape[1]
    hiddenUnits = int(inputUnits * 1.5)
    
    #effettua lo split dei dati di train con quelli di test
    inputMatrix, inputTestMatrix, outputMatrix, outputTestMatrix = train_test_split(inputMatrix, outputMatrix, test_size=test_split, random_state=42)
    
    #Normalizza le matrici di ingresso
    scaler = MinMaxScaler()
    inputMatrix = scaler.fit_transform(inputMatrix)
    inputTestMatrix = scaler.transform(inputTestMatrix)
    
    #log
    #print("matrice di input normalizzata:")
    #print(inputMatrix)
    
    #log
    #print("matrice di output:")
    #print(outputMatrix)

    #log
    print("INIZIO CREAZIONE E ADDESTRAMENTO ANN")
    
    #sovrascrive il grafico predefinito corrente (visto che le invocazioni sono su thread separati)
    with graph.as_default():
   
        #se il modello non esiste lo ricrea altrimenti lo azzera
        #if model is None:     
        #aggiunge lo strato di input ed il primo strato nascosto + una regolarizzazione l2   
        input = Input(shape=(inputUnits,))
        first = Dense(hiddenUnits)(input)
        #first = BatchNormalization()(first)
        first = Activation('tanh')(first)
        first = Dropout(dropout)(first)

        #aggiunge numberHiddenLayer strati nascosti
        hidden = first
        for _ in range(numberHiddenLayers):
            #aggiunge lo strato nascosto
            hidden = Dense(hiddenUnits)(hidden)
            #hidden = BatchNormalization()(hidden)
            hidden = Activation('tanh')(hidden)
            hidden = Dropout(dropout)(hidden)
            
        #aggiunge lo strato di uscita
        output = hidden
        output = Dense(outputUnits)(output)
        #output = BatchNormalization()(output)
        output = Activation('softmax')(output)
         
        #crea il modello
        model = Model(inputs=input, outputs=output)

        #compila la rete neurale
        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        #addestra la rete neurale
        model.fit(inputMatrix, outputMatrix, shuffle=True, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=1)

        #stampa il risultato della valutazione del modello
        printScores(inputTestMatrix, outputTestMatrix)

        #salva lo scaler ed il modello
        joblib.dump(scaler, 'tmp/scaler.pkl')
        model.save('tmp/model.json')
        model.save_weights('tmp/model_weights.h5')
    
    #log
    print("FINE CREAZIONE E ADDESTRAMENTO ANN")


#crea la matrice di input in base alle scansioni passate in ingresso
def makeInputMatrixFromScans(wifiScans):
    
    global wifiMapEncode
    
    #ricostruisce le mappe se necessario
    if wifiMapEncode is None: 
        wifiMapEncode = json.load(open('tmp/wifi_map.json'))
    
    #inizializza a zero la matrice di ingresso 
    inputMatrix = np.zeros((1, len(wifiMapEncode)))    
    
    #effettua un ciclo sulle scansioni passate in ingresso
    for wifiScan in wifiScans:
        
        #ottiene i dettagli della singola scansione
        wifiName = wifiScan["wifiName"]
        wifiLevel = wifiScan["wifiLevel"]
        #print("wifi name: ", wifiName, " level: ", wifiLevel)

        #se la wifiName non e' tra quelle utilizzate per il training, la salta        
        if wifiName not in wifiMapEncode: continue
            
        #ottiene l'indice della matrice di input corrispondente al wifiName
        columnIndex = wifiMapEncode[wifiName]
    
        #popola l'elemento columnIndex dell'inputMatrix con il valore wifiLevel
        inputMatrix[0, columnIndex] = wifiLevel
    
    #torna la matrice di input
    return inputMatrix
    

#effettua una predizione
def predictArea(inputMatrix):
    
    global model, graph, scaler, areaMapDecode
    
    #carica il graph di default
    if graph is None:
        graph = tf.get_default_graph()

    #ricostruisce la rete dai files, se necessario
    if (scaler is None) or (model is None) or (areaMapDecode is None): 
        scaler = joblib.load('tmp/scaler.pkl') 
        model = load_model('tmp/model.json')
        model.load_weights('tmp/model_weights.h5')
        areaMapDecode = json.load(open('tmp/area_map.json'))
    
    #log
    #print("matrice di input:")
    #print(inputMatrix)
    
    #Normalizza la matrice di ingresso
    inputMatrix = scaler.transform(inputMatrix)
    
    #predispone la matrice di uscita
    outputPredictMatrix = np.zeros((1, len(areaMapDecode)))
    
    #log
    print("matrice di input normalizzata:")
    print(inputMatrix)
    
    #sovrascrive il grafico predefinito corrente (visto che le invocazioni sono su thread separati)
    with graph.as_default():
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
    
    #torna l'area con maggiore probabilita'
    return predictArea


#stampa i punteggi e la confusion matrix
def printScores(X_test, Y_test):
    
    global model

    #valuta il modello e stampa lo score
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    
    #prepara i dati per gli altri scores
    Y_pred = model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    
    #calcola gli altri scores
    accuracy_score_val = accuracy_score(Y_test, Y_pred)
    precision_score_val = precision_score(Y_test, Y_pred, average='weighted') # tp / (tp + fp)
    recall_score_val = recall_score(Y_test, Y_pred, average='weighted') # tp / (tp + fn)
    f1_score_val = f1_score(Y_test, Y_pred, average='weighted')

    #stampa gli scores
    print("accuracy_score: %0.4f" % accuracy_score_val)
    print("precision_score: %0.4f" % precision_score_val)
    print("recall_score: %0.4f" % recall_score_val)
    print("f1_score: %0.4f" % f1_score_val)
    
    #calcola e stampa la confusion matrix
    cm = confusion_matrix(Y_test, Y_pred)
    print("confusion matrix:")
    print(cm)
    