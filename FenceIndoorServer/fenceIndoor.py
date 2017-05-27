#install packages:
#pip install flask
#pip install pymongo
#pip install petl
#pip install tensorflow
#pip install keras

from flask import Flask, request
import commonEngine
import dbEngine
import annEngine

#inizializza l'interfaccia rest
app = Flask(__name__)


#ping
@app.route('/ping', methods=['GET'])
def ping():
    return "it works!", 200, {'ContentType':'application/json'} 

	
#inizializza la struttura del database
@app.route('/init', methods=['GET'])
def init():
    dbEngine.clearAndInitDb()
    return "", 200, {'ContentType':'application/json'} 
	

#ritorna la lista delle aree
@app.route("/getAreaList", methods=['GET'])
def getAreaList():
    print("invocato metodo getAreaList");
    
    #ottiene la lista delle aree dal database
    areaList = dbEngine.getAreaListFromDb()
		
    #trasforma la lista di dictionary in stringa e torna l'output
    return commonEngine.json2Str(areaList), 200, {'ContentType':'application/json'} 

	
#acquisisce i dati
@app.route('/sendData/<areaId>', methods=['POST'])
def sendData(areaId):
    print("invocato metodo sendData con areaId: ", areaId);
    
    #trasforma il bodyrequest in json
    inputJson = commonEngine.getJsonFromBodyRequest(request)
    
    #salva le scansioni wifi sul database
    dbEngine.saveWifiScansToDb(areaId, inputJson)
	
    #torna la risposta
    return "", 200, {'ContentType':'application/json'} 

	
#avvia il training
@app.route('/training', methods=['GET'])
def training():
    print("invocato metodo training");
	
    #costruisce i dati
    X,y = annEngine.makeDataFromDbByETL()
    
    #crea la ANN
    annEngine.buildAndFitAnn(X, y)
    
    #torna la risposta
    return "", 200, {'ContentType':'application/json'} 

	
#effettua una predict
@app.route('/predict', methods=['POST'])
def predict():
    print("invocato metodo predict");

    #trasforma il bodyrequest in json
    inputJson = commonEngine.getJsonFromBodyRequest(request)
    
    #effettua un ciclo sulle scansioni passate in ingresso
    for wifiScan in inputJson:
        
        #ottiene i dettagli della singola scansione
        name = wifiScan["name"]
        level = wifiScan["level"]
        print("wifi name: ", name, " level: ", level)
        
    #effettua una predict dell'area
    X = annEngine.makeInputMatrixFromScans(inputJson)
    area = annEngine.predictArea(X)
    
    #trasforma il json di risposta in stringa e torna l'output
    return commonEngine.json2Str(area), 200, {'ContentType':'application/json'} 

	
#main
if __name__ == '__main__':
    #avvia il server in ascolto
    app.run(
        host=commonEngine.getCfg('server', 'address'), 
        port=commonEngine.getCfg('server', 'port'), 
        debug=False
    )
