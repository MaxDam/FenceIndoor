
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
#import petl as etl


#rete neurale artificiale
classifier = None


#costruisce la matrice di input ed il vettore di output 
#in base ai dati presenti sul db utilizzando i flussi ETL
def makeDataFromDbByETL():
    
    #TODO da fare sfruttando gli ETL
    
    inputMatrix = np.zeros(15,100, dtype=np.int)
    outputVector = np.zeros(3, dtype=np.int)
    return inputMatrix, outputVector
    

#costruisce una rete neurale con keras
def buildAndFitAnn(inputMatrix, outputVector):
    
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
    
    #1) ottiene l'elenco delle wifi acquisite  nelle scansioni
    #db.wifiScans.aggregate([ { $group: {"_id":"$wifiName", count:{$sum:1}} } ])
    
    #2) assegna alla variabile <wifiCount> il numero di wifi acquisite
    
    #3) crea un dictionary <wifiMap> dove associa ogni <wifiName> con un numero sequenziale
    #rappresentante la colonna della matrice di input
    
    #4) ottiene le aree dal database
    #db.aree.find()
    
    #5) itera le aree per ottenere dei totali:
    #salva nella variabile <sumScans> la sommatoria dei <lastScanId> delle aree
    #e salva nella variabile <areaCount> il numero totale delle aree
    
    #6) crea una matrice <inputMatrix> fatta di <wifiCount> colonne 
    #e di <sumScans> righe, con valori tutti a zero
    
    #7) crea un vettore <outputVector> fatto di <sumScans> righe, con valori tutti a zero
    
    #8) scorre le aree e per ognuna acquisisce i campi <area> e <lastScanId>, 
    #e mantiene un contatore di righe matrice <rowIndex>
    
    #9) per ogni area effettua un sotto-ciclo da 1 a <lastScanId>
    
    #10) inserisce in <outputVector>[<rowIndex>] = <area>
    
    #10) per ogni iterazione del sotto-ciclo incrementa <rowIndex>
    #e ottiene l'elenco delle scansioni 
    #db.wifiScans.find({ area:'<area>', scanId: <scanId> })
    
    #10) usa il dictionary <wifiMap> ottenere un contatore di colonna <colIndex> 
    #dalla <wifiName> corrente:
    #<colIndex> = <wifiMap>[<wifiName>]
    
    #11) assegna il livello della wifi corrente alla matrice di input
    #considerando i contatori di riga e colonna <rowIndex>,<colIndex>:
    #<inputMatrix>[<rowIndex>,<colIndex>] = <wifiLevel>
    
    #12) finita di comporre la matrice, crea la ANN con:
    #un numero di neuroni di input uguale a <wifiCount>,
    #un numero di neuroni di output uguale a <areaCount>,
    #un numero di neuroni hidden proporzionale all'input ed all'output
    
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

