
from pymongo import MongoClient
from bson.objectid import ObjectId
import commonEngine as com


#si collega al mongodb
clientDb = MongoClient(com.getCfg('database', 'url'))
db = clientDb[com.getCfg('database', 'name')]


#inizializza il database
def clearAndInitDb():

    #cancella le collections
    db.aree.drop()
    db.wifiScans.drop()
    
    #crea le aree
    db.aree.insert_one({"area": "area 1", "lastScanId": 0})
    db.aree.insert_one({"area": "area 2", "lastScanId": 0})
    db.aree.insert_one({"area": "area 3", "lastScanId": 0})


#ritorna un json contenente tutte le aree prese dal database
def getAreaListFromDb():
    
    #ottiene tutte le aree dal database
    cursor = db.aree.find()
	
    #scorre i dati..
    areaList = []
    for areaDoc in cursor:
        #ottiene il dictionary dal documento del database
        area = com.bson2Json(areaDoc)

        #modifica il campo id del dictionary
        area["id"] = area["_id"]["$oid"]
        area.pop('_id', None)
		
        #appende i dati alla lista finale
        areaList.append(area)
    
    return areaList


#salva le scansioni wifi nel database
def saveWifiScansToDb(areaId, inputJson):
    
    #ottiene il record dell'area interessata in base alla chiave primaria
    areaDoc = db.aree.find_one({"_id" : ObjectId(areaId)})
    
    #trasforma il record ottenuto in json
    area = com.bson2Json(areaDoc)
    
    #ottiene il conteggio delle scansioni dal json dell'area
    areaName = area["area"]
    scanId = area["lastScanId"]
	
    #incrementa il numero di scansione
    scanId = scanId + 1
		
    #effettua un ciclo sulle scansioni passate in ingresso
    print("itera le scansioni per l'area: ", areaName)
    for wifiScan in inputJson:
        
        #ottiene i dettagli della singola scansione
        wifiName = wifiScan["wifiName"]
        wifiLevel = wifiScan["wifiLevel"]
        print("wifi name: ", wifiName, " level: ", wifiLevel)
        
        #inserisce la singola scansione nel database
        db.wifiScans.insert_one({"scanId": scanId, "area": areaName, "wifiName": wifiName, "wifiLevel": wifiLevel})
        
    #effettua l'update dell'area
    db.aree.update({"_id" : ObjectId(areaId)}, {"area": areaName, "lastScanId": scanId})

