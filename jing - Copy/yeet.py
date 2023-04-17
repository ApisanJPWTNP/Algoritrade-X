import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
def setupdb():
    cred_obj = firebase_admin.credentials.Certificate('stocks-bb600-firebase-adminsdk-294ak-dfa9afb151.json')
    firebase_admin.initialize_app( cred_obj, {'databaseURL': 'https://stocks-bb600-default-rtdb.asia-southeast1.firebasedatabase.app/'})
def yeettopathdb(Stockname,tdlist):
    ref = db.reference(f'py/users/{Stockname}')
    # tdlist=tdlist.to_json(orient="index")
    ref.set(tdlist)
def getdatadb(Stockname):
    ref = db.reference(f'py/users/{Stockname}')
    datajson=ref.get()
    # datajson = pd.read_json(datajson, orient ='index')
    return datajson
def deletefiledb(Stockname):
    ref = db.reference(f'py/users/{Stockname}')
    ref.delete()
def closeconnection():
    firebase_admin.delete_app(firebase_admin.get_app())