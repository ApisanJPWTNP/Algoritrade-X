# calculate the output of the automated system

def check(Name):
    import pyrebase
    firebaseConfig = {
    'apiKey': "AIzaSyB0KAUat7kWBa9YmY7_5Q090MtIHfVZgp8",
    'authDomain': "stocks-bb600.firebaseapp.com",
    'databaseURL': "https://stocks-bb600-default-rtdb.asia-southeast1.firebasedatabase.app",
    'projectId': "stocks-bb600",
    'storageBucket': "stocks-bb600.firebasestorage.app",
    'messagingSenderId': "808360998402",
    'appId': "1:808360998402:web:354a98f8cb4d3ea4bb991b",
    'measurementId': "G-ZT0MZV9XJ1"
    }
    firebase= pyrebase.initialize_app(firebaseConfig)
    auth= firebase.auth()
    db=firebase.database()
    storage= firebase.storage()
    import findindi 
    import pandas as pd
    import numpy as np
    global signal,buy,sell,dfo,sellbuy,ret,purebh
    amount,ret=0,0
    dfo=findindi.c
    sl1=dfo['volume_prof_signal']
    sl2=dfo['macd_crossover_signal']
    sl3=dfo['psar_adx_signal']
    sl4=dfo['renko_adx_signal']
    sl5=dfo['aroon_kp_signal']
    sl6=dfo['macd_overzero_signal']
    sl7=dfo['ma5_10_signal']
    sl8=dfo['cdc_signal']
    sl9=dfo['bb_rsi_signal']
    sl10=dfo['rsi_adx_signal']
    sl11=dfo['adj_close_price']
    wtd=[0.121529716,
        0.083378998,
        0.16092178499999998,
        0.11823988300000002,
        0.11989504900000003,
        0.054828644,
        0.065031622,
        0.10165256999999998,
        0.08979344000000002,
        0.184728292,
        0.19600576200000003]
    w=wtd[0:10]
    td = wtd[10]
    # for i in range(10):
    #     w.append(0.1)
    wtd=db.child("stockweight").child(Name).get().val()
    if wtd is not None:
        w=wtd[0:10]
        td = wtd[10]
    mon=100000
    buy=[]
    sell=[]
    signala=0
    signal = []
    stockbuy=mon//sl11[0]
    purebh=((sl11[-1]*stockbuy)-100000)/1000
    for i in range(len(dfo)):
        decision_d= ((w[0]*sl1[i]+w[1]*sl2[i]+w[2]*sl3[i]+w[3]*sl4[i]+w[4]*sl5[i]+w[5]*sl6[i]+w[6]*sl7[i]+w[7]*sl8[i]+w[8]*sl9[i]+w[9]*sl10[i])/sum(w))
        if decision_d>td and signala!=1:
            signala=1
            buy.append(sl11[i])
            signal.append(signala)
            sell.append(np.nan)
            sumall=mon//dfo['adj_close_price'][i]
            amount+=sumall
            mon-=sumall*dfo['adj_close_price'][i]*1.002
        elif decision_d<-td and signala!=-1:
            signala=-1
            buy.append(np.nan)
            signal.append(signala)
            sell.append(sl11[i])
            mon+=amount*dfo['adj_close_price'][i]*0.998
            amount=0
        else:
            buy.append(np.nan)
            sell.append(np.nan)
            signal.append(0)
    ret=((mon+amount*dfo['adj_close_price'][len(dfo)-1])-100000)/1000
    rai=dfo.drop(columns=['close_price','adx_signal','macd_crossover_signal','ma30_200_signal','ma50_200_signal','ma30_50_signal','cdc_signal','sar_signal','roc_signal','obv_signal','rsi_signal'])
    rai['buy']=buy
    rai['sell']=sell
    sellbuy=rai.reset_index()
    global result 
    if signal[-1]==1:
        result="BUY"
    elif signal[-1]==-1:
        result="SELL"
    elif signal[-1]==0:
        result="HOLD"
    else:
        result="ERROR!"
    purebh=round(purebh, 2)
    ret=round(ret, 2)
    return sellbuy,result,purebh,ret