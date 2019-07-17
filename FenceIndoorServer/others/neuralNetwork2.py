import json
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib 
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Lambda, Input, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping
from keras.losses import mse, binary_crossentropy
from keras import backend as K
import dataLayer as dl
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
#%matplotlib inline

#set del seed per la randomizzazione
np.random.seed(1671)

'''
#init
getData = GetData()
ae = Autoencoder()
fc = FullyConnectionLayer()

#training
X, Y = getData.getTrainingData()
#X, Y = getData.getTrainingDataFromFile()
getData.plotTrain(X, Y)
getData.confusionMatrix(fc, X, Y)
scoreAE = ae.buildAndFit(X)
ae.plotTrain()
print('AE test score:', scoreAE[0])
print('AE test accuracy:', scoreAE[1])
X = ae.predict(X)
scoreANN = fc.buildAndFit(X, Y)
print('ANN test score:', scoreANN[0])
print('ANN test accuracy:', scoreANN[1])
fc.plotTrain()
getData.save()
ae.save()
fc.save()

#predict
wifiScans = []
for item in inputJson:          
    if(len(wifiScans) == 0):
        wifiScans = item
    else:
        wifiScans = np.vstack((wifiScans, item))
getData.load()
ae.load()
fc.load()
X = getData.getInputData(wifiScans)
getData.plotInput(X)
X = ae.predict(X)
predictions = fc.predict(X);
area = getData.getArea(predictions)
'''


#classe che gestisce i dati (training e scansioni wifi)
class GetData(object):

    def __init__(self):
        #file che conterranno le dictionary e lo scaler
        self.wifiMapFile = 'tmp/wifiMap.json' 
        self.areaMapFile = 'tmp/areaMap.json' 
        self.scalerFile = 'tmp/scaler.pkl'
        self.inputMapFile = 'tmp/X.csv'
        self.outputMapFile = 'tmp/Y.csv'

        #è una dictionary ogni <wifiName> con un numero sequenziale rappresentante la colonna della matrice di input
        self.wifiMapEncode = {}

        #è una dictionary ogni numero sequenziale rappresentante la colonna della matrice di input associa il <wifiName> 
        self.wifiMapDecode = {}

        #areaMapEncode una dictionary che dato il nome dell'area ritorna la posizione dell'area
        self.areaMapEncode = {}

        #areaMapDecode una dictionary che data la posizione dell'area ritorna i dettagli dell'area
        self.areaMapDecode = {}

        #scaler
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        #self.scaler = StandardScaler()

    #torna i dati di training dal db
    def getTrainingData(self):
        #log
        print("INIZIO PREPARAZIONE DATI")

        #step 1 
        #ottiene l'elenco delle wifi acquisite  nelle scansioni, e le itera per:
        # - assegnare a <wifiCount> il numero totale delle wifi acquisite
        # - associare con <wifiMap> ogni <wifiName> con un numero sequenziale rappresentante la colonna della matrice di input
        print("acquisizione delle wifi scansionate..")
        wifiCount = 0 
        wifiList = dl.getWifiListFromDb()
        for wifi in wifiList:
            wifiName = wifi['wifiName']
            self.wifiMapEncode[wifiName] = wifiCount
            self.wifiMapDecode[str(wifiCount)] = wifiName
            wifiCount = wifiCount + 1
        print("scansionate ", wifiCount, " wifi")

        #step 2
        #ottiene le aree dal database, e le itera per:
        # - assegnare a <scanCount> la sommatoria dei <lastScanId> delle aree
        # - assegnare ad <areaCount> il numero totale delle aree
        # - associare con <areaMapEncode> il nome dell'area con un numero sequenziale rappresentante la colonna della matrice di output
        # - associare con <areaMapDecode> il numero sequenziale rappresentante la colonna della matrice di output con i dettagli dell'area
        print("acquisizione delle aree..")
        scanCount = 0
        areaCount = 0
        areaList = dl.getAreaListFromDb()
        for area in areaList:
            scanCount = scanCount + area['lastScanId']
            areaName = area['area']
            if areaName == '': continue
            self.areaMapEncode[areaName] = areaCount
            self.areaMapDecode[str(areaCount)] = area
            areaCount = areaCount + 1
        print("predisposte ", areaCount, " aree")

        #step 3
        # - crea una matrice <inputMatrix> fatta di <scanCount> righe e di <wifiCount> colonne, con valori tutti a zero
        # - crea una matrice <outputMatrix> fatta di <scanCount> righe e di <areaCount> colonne, con valori tutti a zero
        X = np.zeros((scanCount, wifiCount))
        Y = np.zeros((scanCount, areaCount))

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
            columnIndex = self.areaMapEncode[areaName]
            Y[rowIndex, columnIndex] = 1.0

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
                columnIndex = self.wifiMapEncode[wifiName]
                wifiLevel = scan["wifiLevel"] 
                X[rowIndex, columnIndex] = wifiLevel

            rowIndex = rowIndex + 1
        print("preparate le matrici con ", len(areaScanList), " scansioni")

        #scala i dati (addestrando lo scaler)
        X = X.astype('float32')
        Y = Y.astype('float32')
        X = self.scaler.fit_transform(X)

        #log
        print("FINE PREPARAZIONE DATI")

        #torna le matrici di input e output
        return X, Y

    #carica i dati di train dai files
    def getTrainingDataFromFile(self):        
        #ottiene i dati dai files csv
        dfX = pd.read_csv(self.inputMapFile)
        dfY = pd.read_csv(self.outputMapFile)
        X = dfX.iloc[:,:].values.astype(np.int32)
        Y = dfY.iloc[:,:].values.astype(np.int32)
        
        #ottiene i json dai files
        self.wifiMapEncode = json.load(open(self.wifiMapFile))
        self.areaMapDecode = json.load(open(self.areaMapFile))
        self.wifiMapDecode = {str(v): k for k, v in self.wifiMapEncode.items()}
        self.areaMapEncode = {v['area']: int(k) for k, v in self.areaMapDecode.items()}

        #scala i dati (addestrando lo scaler)
        X = X.astype('float32')
        Y = Y.astype('float32')
        X = self.scaler.fit_transform(X)
        
        return X, Y
        
    #torna i dati di input date le scansioni wifi 
    def getInputData(self, wifiScans):
        #inizializza a zero la matrice di input
        X = np.zeros((len(wifiScans), len(self.wifiMapEncode)))

        #effettua un ciclo sulle scansioni passate in ingresso al metodo
        #wifiScans (insieme di letture) che contengono wifiScan (singola lettura) che contiene wifiMap (con elementi nome, valore)
        rowCount = 0
        for wifiScan in wifiScans:
            for wifiMap in wifiScan:
                #ottiene i dettagli della singola scansione
                wifiName = wifiMap["wifiName"]
                wifiLevel = wifiMap["wifiLevel"]
                
                #se la wifiName e' tra quelle utilizzate per il training..        
                if wifiName in self.wifiMapEncode:

                    #ottiene l'indice della colonna corrispondente al wifiName
                    columnIndex = self.wifiMapEncode[wifiName]
                
                    #popola l'elemento rowCount, columnIndex della matrice di input con il valore wifiLevel
                    X[rowCount, columnIndex] = wifiLevel

            rowCount += 1

        #scala i dati della matrice di input      
        X = X.astype('float32')
        X = self.scaler.transform(X)

        #torna la matrice di input
        return X

    def getArea(self, predictions):
        '''
        #ottiene gli id delle previsioni massime
        max_predictions_id = np.argmax(predictions, axis=1)
        print("matrice indici di previsione massima:")
        print(max_predictions_id)

        #ottiene l'id dell'area predetta con maggiore frequenza
        max_frequency_area_id = np.argmax(np.bincount(max_predictions_id))
        print("indice dell'area predetta con maggiore frequenza:")
        print(max_frequency_area_id)

        #attiene il nome dell'area e lo ritorna
        max_frequency_area = self.areaMapDecode[str(max_frequency_area_id)]
        return max_frequency_area
        '''
        sum_predictions = np.sum(predictions, axis=0)
        max_sum_predictions_id = np.argmax(sum_predictions)
        max_prediction_area = self.areaMapDecode[str(max_sum_predictions_id)]
        return max_prediction_area

    def saveData(self, X, Y):    
        np.savetxt(self.inputMapFile, X, delimiter=",",fmt='%.0f')
        np.savetxt(self.outputMapFile, Y, delimiter=",",fmt='%.0f')

    def save(self):
        json.dump(self.wifiMapEncode, open(self.wifiMapFile,'w'))
        json.dump(self.areaMapDecode, open(self.areaMapFile,'w'))
        joblib.dump(self.scaler, self.scalerFile)
    
    def load(self):
        self.wifiMapEncode = json.load(open(self.wifiMapFile))
        self.areaMapDecode = json.load(open(self.areaMapFile))
        self.scaler = joblib.load(self.scalerFile) 

    def plotTrain(self, X, Y):
        areas = {}
        for areaIndex in range(Y.shape[1]):
            for rowIndex in range(Y.shape[0]):
                if Y[rowIndex, areaIndex] == 1:
                    if str(areaIndex) in areas:
                        areas[str(areaIndex)] = np.vstack((areas[str(areaIndex)], X[rowIndex, :]))
                    else:
                        areas[str(areaIndex)] = np.reshape(X[rowIndex, :], (1,X.shape[1]))

        for areaIndex, value in areas.items():
            labels = []
            plotHandles = []
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(111)
            for wifiIndex in range(0, value.shape[1]):
                ph, = ax1.plot(value[wifiIndex], label=wifiIndex)
                plotHandles.append(ph)
                wifiName = self.wifiMapDecode[str(wifiIndex)]
                labels.append(wifiName)
            area = self.areaMapDecode[str(areaIndex)]
            plt.title(area['area'])
            plt.legend(plotHandles, labels, loc='upper left', ncol=1, bbox_to_anchor=(1, 1))

    def plotInput(self, X):
        labels = []
        plotHandles = []
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        for wifiIndex in range(0, X.shape[1]):
            ph, = ax1.plot(X[wifiIndex], label=wifiIndex)
            plotHandles.append(ph)
            wifiName = self.wifiMapDecode[str(wifiIndex)]
            labels.append(wifiName)
        plt.title("inputs")
        plt.legend(plotHandles, labels, loc='upper left', ncol=1, bbox_to_anchor=(1, 1))

    
#Autoencoder
class Autoencoder(object):

    def __init__(self):
        self.autoencoderModelFile = 'tmp/ae.json'
        self.autoencoderWeightFile= 'tmp/ae.h5'
        self.verbose=1

        #hyperparameters
        self.use_dropout = True
        self.use_autodimension = True
        self.encoding_dim = 32
        #self.encoding_dim = 16
        self.hidden_dim = self.encoding_dim * 2
        self.dropout = 0.3
        self.batch_size=32
        self.epochs=200
        self.shuffle=False
        self.validation_split=0.3
        self.lr=0.001
        
    def buildAndFit(self, X):
        #log
        print("ADDESTRAMENTO AUTOENCODER")

        #attrubuisce le dimensioni degli strati in automatico in base alla dimensione dell'input
        if self.use_autodimension:
            self.encoding_dim = int(X.shape[1] / 4)
            self.hidden_dim = int(X.shape[1] / 2)

        #crea il modello autoencoder
        in_out_neurons_number = X.shape[1]
        inputLayer = Input(shape=(in_out_neurons_number, ))
        encoderLayer = Dense(self.hidden_dim, activation="tanh")(inputLayer)
        if self.use_dropout:
            encoderLayer = Dropout(self.dropout)(encoderLayer)
        encoderLayer = Dense(self.encoding_dim, activation="tanh")(encoderLayer)
        decoderLayer = Dense(self.hidden_dim, activation='tanh')(encoderLayer)
        if self.use_dropout:
            decoderLayer = Dropout(self.dropout)(decoderLayer)
        decoderLayer = Dense(in_out_neurons_number, activation='sigmoid')(decoderLayer)
        self.model = Model(inputs=inputLayer, outputs=decoderLayer)
        self.encoder = Model(inputLayer, encoderLayer)

        #inizializza la tensorboard per l'autoencoder (http://localhost:6006/)
        tensorboard = TensorBoard(log_dir='./logs/autoencoder', histogram_freq=0, write_graph=True, write_images=True)

        #compila l'autoencoder
        optimizer = Adam(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

        #addestra l'autoencoder
        self.history = self.model.fit(X, X, epochs=self.epochs, batch_size=self.batch_size, shuffle=self.shuffle, validation_split=self.validation_split, verbose=self.verbose, callbacks=[tensorboard])
        
        #torna il punteggio del training
        score = self.model.evaluate(X, X, verbose=0)
        return score

    def predict(self, X):
        return self.encoder.predict(X)

    def save(self):
        self.encoder.save(self.autoencoderModelFile)
        self.encoder.save_weights(self.autoencoderWeightFile)

    def load(self):
        self.encoder = load_model(self.autoencoderModelFile)
        self.encoder.load_weights(self.autoencoderWeightFile)

    def plotTrain(self):
        # summarize history for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

class VarationAutoencoder(object):

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.vaeModelFile = 'tmp/vae.json'
        self.vaeWeightFile = 'tmp/vae.h5'
        self.verbose = 1
        
        #hyperparameters
        self.test_size=0.33
        self.intermediate_dim = 512
        self.batch_size = 128
        self.epochs = 50
        
    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    def sampling(self, args):
        z_mean, z_log_var = args
        # K is the keras backend
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def buildAndFit(self, X):

        #effettua lo split dei dati di train con quelli di test
        X_train, X_test = train_test_split(X, test_size=self.test_size, random_state=42)

        # VAE model = encoder + decoder
        # build encoder model
        input_shape = X.shape[1]
        inputs = Input(shape=(input_shape, ), name='encoder_input')
        x = Dense(self.intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        self.encoder.summary()
        #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(input_shape, activation='sigmoid')(x)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()
        #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

        # instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')

        ##### train model ####    

        #models = (self.encoder, self.decoder)
        #data = (x_test, y_test)

        # VAE loss = mse_loss or xent_loss + kl_loss
        reconstruction_loss = mse(inputs, outputs)
        #reconstruction_loss = binary_crossentropy(inputs, outputs)

        reconstruction_loss *= input_shape
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        self.vae.summary()
        #plot_model(vae, to_file='vae_mlp.png', show_shapes=True)

        # train the autoencoder
        self.history = self.vae.fit(X_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_test, None), verbose=self.verbose)

        #stampa il risultato della valutazione del modello
        score = self.vae.evaluate(X_test, X_test, verbose=0)
        return score
    
    def predict(self, X):
        z = self.encoder.predict(X)[2]
        return z

    def save(self):
        self.encoder.save(self.vaeModelFile)
        self.encoder.save_weights(self.vaeWeightFile)

    def load(self):
        self.encoder = load_model(self.vaeModelFile)
        self.encoder = self.vae.load_weights(self.vaeWeightFile)

    def plotTrain(self):
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


class VarationAutoencoder2(object):

    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.vaeModelFile = 'tmp/vae.json'
        self.vaeWeightFile = 'tmp/vae.h5'
        self.verbose = 1
        
        #hyperparameters
        self.intermediate_dim = 512
        self.batch_size = 128
        self.epochs = 50
        self.validation_split=0.3
        self.test_size=0.33

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def buildAndFit(self, X):
        
        #effettua lo split dei dati di train con quelli di test
        X_train, X_test = train_test_split(X, test_size=self.test_size, random_state=42)

        #build encoder model
        input_shape = X.shape[1]
        inputs = Input(shape=(input_shape, ), name='encoder_input')
        vae_z_in = Dense(self.intermediate_dim, activation='relu')(inputs)
        vae_z_mean = Dense(self.latent_dim)(vae_z_in)
        vae_z_log_var = Dense(self.latent_dim)(vae_z_in)
        # Using the Lambda Keras class around the sampling function we created above
        vae_z = Lambda(self.sampling)([vae_z_mean, vae_z_log_var])

        # instantiate encoder model
        self.encoder = Model(inputs, [vae_z_mean, vae_z_log_var, vae_z], name='encoder')
        self.encoder.summary()

        # build decoder model
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        vae_z_out = Dense(self.intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(input_shape, activation='sigmoid')(vae_z_out)

        # instantiate decoder model
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder.summary()

        # instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae_mlp')

        def vae_r_loss(y_true, y_pred):
            y_true_flat = K.flatten(y_true)
            y_pred_flat = K.flatten(y_pred)
            return 10 * K.mean(K.square(y_true_flat - y_pred_flat), axis = -1)
        # Defining the KL divergence loss
        def vae_kl_loss(y_true, y_pred):
            return - 0.5 * K.mean(1 + vae_z_log_var - K.square(vae_z_mean) - K.exp(vae_z_log_var), axis = -1)
        # Defining the total VAE loss, summing the MSE and KL losses
        def vae_loss(y_true, y_pred):
            return vae_r_loss(y_true, y_pred) + vae_kl_loss(y_true, y_pred)
        # Compiling the whole model with the RMSProp optimizer, the vae loss and custom metrics
        self.vae.compile(optimizer='rmsprop', loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss])

        ##### train model ####    
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
        callbacks_list = [earlystop]
        self.history = self.vae.fit(X_train, X_train, shuffle=True, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split, verbose=self.verbose, callbacks=callbacks_list)

        #stampa il risultato della valutazione del modello
        score = self.vae.evaluate(X_test, X_test, verbose=0)
        return score
    
    def predict(self, X):
        z = self.encoder.predict(X)[2]
        return z

    def save(self):
        self.encoder.save(self.vaeModelFile)
        self.encoder.save_weights(self.vaeWeightFile)

    def load(self):
        self.encoder = load_model(self.vaeModelFile)
        self.encoder = self.vae.load_weights(self.vaeWeightFile)

    def plotTrain(self):
        # summarize history for r_loss
        plt.plot(self.history.history['vae_r_loss'])
        plt.plot(self.history.history['val_vae_r_loss'])
        plt.title('model r loss')
        plt.ylabel('r loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for kl_loss
        plt.plot(self.history.history['vae_kl_loss'])
        plt.plot(self.history.history['val_vae_kl_loss'])
        plt.title('model kl loss')
        plt.ylabel('kl loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

#ANN
class FullyConnectionLayer(object):
    
    def __init__(self):
        self.fcModelFile = 'tmp/fc.json'
        self.fcWeightFile= 'tmp/fc.h5'
        self.verbose=1
        
        #hyperparameters
        self.use_batchnorm = False
        self.use_dropout = True
        self.numberHiddenLayers = 3
        self.dropout_input = 0.1
        self.dropout = 0.3
        self.batch_size=128
        self.epochs = 200
        self.test_size=0.33
        self.lr=0.001
    
    def buildAndFit(self, X, Y):
        #log
        print("ADDESTRAMENTO ANN")

        #calcola: 
        #numero di neuroni di input in base al numero di colonne di inputMatrix
        #numero di neuroni di output in base al numero di colonne di outputMatrix
        #un numero di neuroni hidden proporzionale all'input ed all'output
        inputUnits = X.shape[1]
        outputUnits = Y.shape[1]
        hiddenUnits = int(inputUnits * 1.5)

        #effettua lo split dei dati di train con quelli di test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.test_size, random_state=42)

        #aggiunge lo strato di input ed il primo strato nascosto + una regolarizzazione l2   
        input = Input(shape=(inputUnits,))
        first = input
        if self.use_batchnorm:
            first = BatchNormalization()(first)
        first = Activation('elu')(first)
        if self.use_dropout:
            first = Dropout(self.dropout_input)(first)

        #aggiunge numberHiddenLayer strati nascosti
        hidden = first
        for _ in range(self.numberHiddenLayers):
            #aggiunge lo strato nascosto
            hidden = Dense(hiddenUnits)(hidden)
            if self.use_batchnorm:
                hidden = BatchNormalization()(hidden)
            hidden = Activation('elu')(hidden)
            if self.use_dropout:
                hidden = Dropout(self.dropout)(hidden)

        #aggiunge lo strato di uscita
        output = Dense(outputUnits)(hidden)
        if self.use_batchnorm:
            output = BatchNormalization()(output)
        output = Activation('softmax')(output)
        self.model = Model(inputs=input, outputs=output)

        #inizializza la tensorboard per l'ann
        tensorboard = TensorBoard(log_dir='./logs/ann', histogram_freq=0, write_graph=True, write_images=True)

        #compila la rete neurale
        optimizer = Adam(lr=self.lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        #addestra la rete neurale
        self.history = self.model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs, validation_data=(X_test, Y_test), verbose=self.verbose, callbacks=[tensorboard])

        #stampa il risultato della valutazione del modello
        score = self.model.evaluate(X_test, Y_test, verbose=0)
        return score

    def predict(self, X):
        #effettua la previsione
        predictions = self.model.predict(X)
        return predictions

    def save(self):
        self.model.save(self.fcModelFile)
        self.model.save_weights(self.fcWeightFile)

    def load(self):
        self.model = load_model(self.fcModelFile)
        self.model.load_weights(self.fcWeightFile)

    def plotTrain(self):
        # summarize history for accuracy
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        
    def confusionMatrix(self, X_test, Y_test):
        #prepara i dati
        Y_pred = self.model.predict(X_test)
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
        
        #ottiene e stampa la confusion matrix
        cm = confusion_matrix(Y_test, Y_pred)
        df_cm = pd.DataFrame(cm, 
                             index = [i for i in range(cm.shape[0])],
                             columns = [i for i in range(cm.shape[1])])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
        