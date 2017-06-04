
#funzioni di utilita' dao con accesso al database mongodb

import os
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId
import commonEngine as com


#si collega al mongodb e setta il database
if(com.getCfg('docker')):
	clientDb = MongoClient(os.environ['DB_PORT_27017_TCP_ADDR'], 27017)
else:
	clientDb = MongoClient(com.getCfg('database', 'url'))

db = clientDb[com.getCfg('database', 'name')]


#inizializza il database
def clearAndInitDb():

    #cancella le collections
    db.areas.drop()
    db.wifiScans.drop()
    
    #crea le aree
    db.areas.insert_one({"area": "area 1", "lastScanId": 0})
    db.areas.insert_one({"area": "area 2", "lastScanId": 0})
    db.areas.insert_one({"area": "area 3", "lastScanId": 0})


#ritorna un json contenente tutte le aree prese dal database
def getAreaListFromDb():
    
    #ottiene tutte le aree dal database
    cursor = db.areas.find()
	
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


#aggiunge un'area nel database
def addAreaToDb(area):
    
    #ottiene il nome dell'area
    areaName = area['area']
    
    #inserisce l'area nel db
    db.areas.insert_one({"area": areaName, "lastScanId": 0})


#cancella un'area dal database
def deleteAreaToDb(area):
    
    #rimuove dal db le scansioni per l'area da cancellare
    areaName = area['area']
    db.wifiScans.delete_many({"area": areaName})
    
    #rimuove l'area dal db
    areaId = area['id']
    db.areas.delete_many({"_id": ObjectId(areaId)})


#salva le scansioni wifi nel database
def saveWifiScansToDb(areaId, wifiList):
    
    #ottiene il record dell'area interessata in base alla chiave primaria
    areaDoc = db.areas.find_one({"_id" : ObjectId(areaId)})
    
    #trasforma il record ottenuto in json
    area = com.bson2Json(areaDoc)
    
    #ottiene il conteggio delle scansioni dal json dell'area
    areaName = area["area"]
    scanId = area["lastScanId"]
	
    #incrementa il numero di scansione
    scanId = scanId + 1
		
    #effettua un ciclo sulle scansioni passate in ingresso
    print("itera le scansioni per l'area: ", areaName)
    for wifiScan in wifiList:
        
        #ottiene i dettagli della singola scansione
        wifiName = wifiScan["wifiName"]
        wifiLevel = wifiScan["wifiLevel"]
        print("wifi name: ", wifiName, " level: ", wifiLevel)
        
        #inserisce la singola scansione nel database
        db.wifiScans.insert_one({"scanId": scanId, "area": areaName, "wifiName": wifiName, "wifiLevel": wifiLevel})
        
    #effettua l'update dell'area
    db.areas.update({"_id" : ObjectId(areaId)}, {"area": areaName, "lastScanId": scanId})


#ritorna l'elenco di tutte le wifi uniche acquisite
def getWifiListFromDb():
    
    #ottiene tutte le aree dal database
    cursor = db.wifiScans.aggregate([ { "$group": {"_id":"$wifiName", "count":{"$sum":1}} } ])
	
    #scorre i dati..
    wifiList = []
    for wifiDoc in cursor:
        #ottiene il dictionary dal documento del database
        wifi = com.bson2Json(wifiDoc)

        #modifica il campo id del dictionary
        wifi["wifiName"] = wifi["_id"]
        wifi.pop('_id', None)
		
        #appende i dati alla lista finale
        wifiList.append(wifi)
    
    return wifiList


#torna le associazioni uniche area-scanId prese dalle scansioni
def getAreaAndScanIdListFromDb():
    
    #ottiene tutte le aree dal database
    cursor = db.wifiScans.aggregate([ { "$group": {"_id":{ "area":"$area", "scanId":"$scanId"} , "count":{"$sum":1}} } ])
    
    #scorre i dati..
    resultList = []
    for itemDoc in cursor:
        #ottiene il dictionary dal documento del database
        item = com.bson2Json(itemDoc)

        #modifica il campo id del dictionary
        item["area"]   = item["_id"]["area"]
        item["scanId"] = item["_id"]["scanId"]
        item.pop('_id', None)
		
        #appende i dati alla lista finale
        resultList.append(item)
    
    return resultList


#torna l'elenco di scansioni effettuate per area e scanId
def getScansFromDb(area, scanId):
    
    #ottiene tutte le aree dal database
    cursor = db.wifiScans.find({ "area":area, "scanId": scanId })
    
    #scorre i dati..
    scanList = []
    for scanDoc in cursor:
        #ottiene il dictionary dal documento del database
        scan = com.bson2Json(scanDoc)

        #modifica il campo id del dictionary
        scan.pop('_id', None)
		
        #appende i dati alla lista finale
        scanList.append(scan)
    
    return scanList


#indicizza le scansioni per area e scanId, in modo da avere un accesso veloce
def indexScans():
    db.wifiScans.create_index([("area", pymongo.ASCENDING), ("scanId", pymongo.ASCENDING)])