
from pymongo import MongoClient
from bson import json_util
from bson.objectid import ObjectId
import json
import commonEngine as com

#si collega al mongodb
def getDb():
    clientDb = MongoClient(com.getCfg('database', 'url'))
    return clientDb[com.getCfg('database', 'name')]


#trasforma una stringa in bson
def str2Bson(str):
    return json_util.loads(str)


#trasforma il bson in una stringa
def bson2Str(bsonObj):
    return json_util.dumps(bsonObj)

#trasforma il bson in un json
def bson2Json(bsonObj):
    return json.loads(json_util.dumps(bsonObj))


#inizializza il database
def clearAndInitDb():
    #ottiene l'oggetto per accedere al db
    db = getDb()
    
    #cancella le collections
    db.aree.drop()
    db.wifiScans.drop()
    
    #crea le aree
    db.aree.insert_one({"name": "area 1", "scanCount": 0})
    db.aree.insert_one({"name": "area 2", "scanCount": 0})
    db.aree.insert_one({"name": "area 3", "scanCount": 0})


#ritorna un json contenente tutte le aree prese dal database
def getAreaListFromDb():

    #ottiene l'oggetto per accedere al db
    db = getDb()
    
    #ottiene tutte le aree dal database
    cursor = db.aree.find()
	
    #scorre i dati..
    areaList = []
    for areaDoc in cursor:
        #ottiene il dictionary dal documento del database
        area = bson2Json(areaDoc)

        #modifica il campo id del dictionary
        area["id"] = area["_id"]["$oid"]
        area.pop('_id', None)
		
        #appende i dati alla lista finale
        areaList.append(area)
    
    return areaList


#salva le scansioni wifi nel database
def saveWifiScansToDb(areaId, inputJson):
    
    #ottiene l'oggetto per accedere al db
    db = getDb()
    
    #ottiene il record dell'area interessata in base alla chiave primaria
    areaDoc = db.aree.find_one({"_id" : ObjectId(areaId)})
    
    #trasforma il record ottenuto in json
    area = bson2Json(areaDoc)
    
    #ottiene il conteggio delle scansioni dal json dell'area
    areaName = area["name"]
    scanCount = area["scanCount"]
	
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
    
    return
