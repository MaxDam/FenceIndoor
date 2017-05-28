#install packages:
#pip install flask
#pip install pymongo
#pip install petl
#pip install tensorflow
#pip install keras


#interfacce RESTful

from flask import Flask, request
import commonEngine as com
import dbEngine as dao
import annEngine as ann


#inizializza l'interfaccia rest
app = Flask(__name__)


#ping
@app.route('/ping', methods=['GET'])
def ping():
    return "it works!", 200, {'ContentType':'text/html'} 

	
#inizializza la struttura del database
@app.route('/init', methods=['GET'])
def init():
    try:
        #inizializza il db
        dao.clearAndInitDb()
        
        return "", 200, {'ContentType':'application/json'} 
	 
    except Exception as e:
        
        return str(e), 500, {'ContentType':'text/html'} 


#ritorna la lista delle aree
@app.route("/getAreaList", methods=['GET'])
def getAreaList():

    print("invocato metodo getAreaList");
    
    try:
        #ottiene la lista delle aree dal database
        areaList = dao.getAreaListFromDb()
    		
        #trasforma la lista di dictionary in stringa e torna l'output
        return com.json2Str(areaList), 200, {'ContentType':'application/json'} 
    
    except Exception as e:
        
        return str(e), 500, {'ContentType':'text/html'} 
	

#acquisisce i dati
@app.route('/sendData/<areaId>', methods=['POST'])
def sendData(areaId):
    
    print("invocato metodo sendData con areaId: ", areaId);
    
    try:
        #trasforma il bodyrequest in json
        inputJson = com.bodyRequest2Json(request)
        
        #salva le scansioni wifi sul database
        dao.saveWifiScansToDb(areaId, inputJson)
    	
        #torna la risposta
        return "", 200, {'ContentType':'application/json'} 

    except Exception as e:
        
        return str(e), 500, {'ContentType':'text/html'}  
	

#avvia il training
@app.route('/training', methods=['GET'])
def training():
    
    print("invocato metodo training");
	
    try:
        #costruisce i dati
        X, Y = ann.makeDataFromDb()
        
        #crea la ANN
        ann.buildAndFitAnn(X, Y)
        
        #torna la risposta
        return "", 200, {'ContentType':'application/json'} 

    except Exception as e:
        
        return str(e), 500, {'ContentType':'text/html'} 
    

#effettua una predict
@app.route('/predict', methods=['POST'])
def predict():
    
    print("invocato metodo predict");

    try:
        #trasforma il bodyrequest in json
        inputJson = com.bodyRequest2Json(request)
        
        #effettua una predict dell'area
        X = ann.makeInputMatrixFromScans(inputJson)
        area = ann.predictArea(X)
        
        #trasforma il json di risposta in stringa e torna l'output
        return com.json2Str(area), 200, {'ContentType':'application/json'} 

    except Exception as e:
        
        return str(e), 500, {'ContentType':'text/html'} 
    
	
#main
if __name__ == '__main__':
    #avvia il server in ascolto
    app.run(
        host =com.getCfg('server', 'address'), 
        port =com.getCfg('server', 'port'), 
        debug=com.getCfg('server', 'debug')
    )
