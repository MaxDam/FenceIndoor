#install packages:
#pip install flask
#pip install pymongo
#pip install tensorflow
#pip install keras

from flask import Flask, request
from pymongo import MongoClient
from bson import json_util
from bson.objectid import ObjectId
import json

#parametri di connessione
DB_URL = "mongodb://localhost:27017"
DB_NAME = "fi"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8090

#inizializza l'interfaccia rest
app = Flask(__name__)

#si collega al mongodb
def getDb():
    clientDb = MongoClient(DB_URL)
    return clientDb[DB_NAME]

#ping
@app.route('/ping', methods=['GET'])
def ping():
    return "it works!", 200, {'ContentType':'application/json'} 
	
#inizializza la struttura
@app.route('/init', methods=['GET'])
def init():
    #ottiene l'oggetto per accedere al db
    db = getDb()
    
    #cancella le collections
    db.aree.drop()
    db.wifiScans.drop()
    
    #crea le aree
    db.aree.insert_one({"name": "area 1", "scanCount": 0})
    db.aree.insert_one({"name": "area 2", "scanCount": 0})
    db.aree.insert_one({"name": "area 3", "scanCount": 0})
    return "", 200, {'ContentType':'application/json'} 
	
#ritorna la lista delle aree
@app.route("/getAreaList", methods=['GET'])
def getAreaList():
    print("invocato metodo getAreaList");
    
    #ottiene l'oggetto per accedere al db
    db = getDb()
    
	#ottiene tutte le aree dal database
    cursor = db.aree.find()
	
	#scorre i dati..
    areaJsonList = []
    for areaDoc in cursor:
        #ottiene il dictionary dal documento del database
        areaJson = json.loads(json_util.dumps(areaDoc))

        #modifica il campo id del dictionary
        areaJson["id"] = areaJson["_id"]["$oid"]
        areaJson.pop('_id', None)
		
        #appende i dati alla lista finale
        areaJsonList.append(areaJson)
		
    #trasforma la lista di dictionary in stringa e torna l'output
    return json.dumps(areaJsonList), 200, {'ContentType':'application/json'} 
	
#acquisisce i dati
@app.route('/sendData/<areaId>', methods=['POST'])
def sendData(areaId):
    print("invocato metodo sendData con areaId: ", areaId);
    
    #ottiene l'oggetto per accedere al db
    db = getDb()
    
    #ottiene il bodyrequest in formato stringa
    bodyRequest = request.data.decode('utf8')
    print("request: ", bodyRequest)
    
    #trasforma il bodyrequest in json
    inputJson = json.loads(bodyRequest)
    
	#ottiene il record dell'area interessata in base alla chiave primaria
    areaDoc = db.aree.find_one({"_id" : ObjectId(areaId)})
    
    #trasforma il record ottenuto in json
    areaJson = json.loads(json_util.dumps(areaDoc))
    
    #ottiene il conteggio delle scansioni dal json dell'area
    areaName = areaJson["name"]
    scanCount = areaJson["scanCount"]
	
    #effettua un ciclo sulle scansioni passate in ingresso
    print("itera le scansioni per l'area: ", areaName)
    for wifiScan in inputJson:
        
        #ottiene i dettagli della singola scansione
        wifiName = wifiScan["name"]
        wifiLevel = wifiScan["level"]
        print("wifi name: ", wifiName, " level: ", wifiLevel)
        
        #inserisce la singola scansione nel database
        db.wifiScans.insert_one({"area": areaName, "name": wifiName, "level": wifiLevel})
        
    #incrementa il numero di scansioni
    scanCount = scanCount + 1
		
	#effettua l'update dell'area
    db.aree.update({"_id" : ObjectId(areaId)}, {"name": areaName, "scanCount": scanCount})
	
    #torna la risposta
    return "", 200, {'ContentType':'application/json'} 
	
#avvia il training
@app.route('/training', methods=['GET'])
def training():
    print("invocato metodo training");
	
    #todo inizia il training in base alle scansioni memorizzate nel database
    
    #torna la risposta
    return "", 200, {'ContentType':'application/json'} 
	
#effettua una predict
@app.route('/predict', methods=['POST'])
def predict():
    print("invocato metodo predict");

    #ottiene il bodyrequest in formato stringa
    bodyRequest = request.data.decode('utf8')
    print("request: ", bodyRequest)
	
    #trasforma il bodyrequest in json
    inputJson = json.loads(bodyRequest)
    
    #effettua un ciclo sulle scansioni passate in ingresso
    for wifiScan in inputJson:
        
        #ottiene i dettagli della singola scansione
        name = wifiScan["name"]
        level = wifiScan["level"]
        print("wifi name: ", name, " level: ", level)
		
        #todo acquisisce le scansioni   
        
    #todo effettua una predict dell'area
        
    #prepara la risposta
    jsonOutput = {}
    jsonOutput['name'] = "Non lo so ancora.. abbi un poco di pazienza";
    
    #trasforma il json di risposta in stringa e torna l'output
    return json.dumps(jsonOutput), 200, {'ContentType':'application/json'} 
	
#main
if __name__ == '__main__':
    #avvia il server in ascolto
    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=False)
	