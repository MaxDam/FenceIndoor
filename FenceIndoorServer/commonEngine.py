
import yaml
import json

#acquisisce il file di configurazione
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

#ottiene una proprieta' dalla configurazione
def getCfg(*keys):
    dct = cfg
    for key in keys:
        try:
            dct = dct[key]
        except KeyError:
            return None
    return dct

#data la request torna il json in body request
def getJsonFromBodyRequest(request):
    
    #ottiene il body request
    bodyRequest = request.data.decode('utf8')
    print("request: ", bodyRequest)

    #trasforma il body request in json
    return str2Json(bodyRequest)


#trasforma una stringa in json
def str2Json(str):
    return json.loads(str)


#trasforma il json in una stringa
def json2Str(jsonObj):
    return json.dumps(jsonObj)

