
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

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

