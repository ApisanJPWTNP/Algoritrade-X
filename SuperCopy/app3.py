from flask import Flask,render_template,url_for,request,redirect,jsonify,session
from findindi import *
from checksignal import *

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
from pyrebase import pyrebase
app= Flask(__name__)
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
sucees=False
statussignup=''
statuslogin=''
status=False

from collections import Counter
def count_stock_in_port(athree_l):
    global res_dict,l
    num_stock_inport_l=[]
    for i in athree_l:
        l=[]
        l.append(i[0])
        l.append(i[2])
        num_stock_inport_l.append(l)
    d = {}
    for a in num_stock_inport_l:
        if a[0] in d:
            s = d[a[0]]
            s.append(a[1])
            d[a[0]] = s
        else:
            d[a[0]] = [a[1]]
    return d

graphexwid=450
graphexheight=300
colorret="grey"
colorbh="grey"
findindis("CPALL.BK")
sellbuy,result1,purebh1,ret1=check("CPALL")
fig = px.line(sellbuy, x='Date', y="adj_close_price")
fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
fig.update_layout(title={'text': "CPALL",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
fig.update_layout(width=graphexwid, height=graphexheight)
graphJSON1 = plotly.io.to_json(fig)
if ret1<0:
    colorret1="red"
elif ret1>0:
    colorret1="green"
else:
    colorret1="grey"
if purebh1<0:
    colorbh1="red"
elif purebh1>0:
    colorbh1="green"
else:
    colorbh1="grey"
findindis("ADVANC.BK")
sellbuy,result2,purebh2,ret2=check("ADVANC")
fig = px.line(sellbuy, x='Date', y="adj_close_price")
fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
fig.update_layout(title={'text': "ADVANC",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
fig.update_layout(width=graphexwid, height=graphexheight)
graphJSON2 = plotly.io.to_json(fig)
if ret2<0:
    colorret2="red"
elif ret2>0:
    colorret2="green"
else:
    colorret2="grey"
if purebh2<0:
    colorbh2="red"
elif purebh2>0:
    colorbh2="green"
else:
    colorbh2="grey"
findindis("AOT.BK")
sellbuy,result3,purebh3,ret3=check("AOT")
fig = px.line(sellbuy, x='Date', y="adj_close_price")
fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
fig.update_layout(title={'text': "AOT",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
fig.update_layout(width=graphexwid, height=graphexheight)
graphJSON3 = plotly.io.to_json(fig)
if ret3<0:
    colorret3="red"
elif ret3>0:
    colorret3="green"
else:
    colorret3="grey"
if purebh3<0:
    colorbh3="red"
elif purebh3>0:
    colorbh3="green"
else:
    colorbh3="grey"
findindis("PTT.BK")
sellbuy,result4,purebh4,ret4=check("PTT")
fig = px.line(sellbuy, x='Date', y="adj_close_price")
fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
fig.update_layout(title={'text': "PTT",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
fig.update_layout(width=graphexwid, height=graphexheight)
graphJSON4 = plotly.io.to_json(fig)
if ret4<0:
    colorret4="red"
elif ret4>0:
    colorret4="green"
else:
    colorret4="grey"
if purebh4<0:
    colorbh4="red"
elif purebh4>0:
    colorbh4="green"
else:
    colorbh4="grey"

findindis("SCC.BK")
sellbuy,result,purebh,ret=check("SCC")
fig = px.line(sellbuy, x='Date', y="adj_close_price")
fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
fig.update_layout(title={'text': "SCC",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
fig.update_layout(width=1200, height=500)
graphJSON5 = plotly.io.to_json(fig)


#home
@app.route('/', methods=['POST','GET'])
def home():
    return render_template("index.html")

#login
@app.route('/login', methods=['POST','GET'])
def login():

    global sucees
    sucees =False
    if request.method == "POST":
        user = request.form.get("user")
        password= request.form.get("password")
        try:
            user = auth.sign_in_with_email_and_password(user, password)
            sucees =True
            app.secret_key =user['localId']
            session['ID'] = user['localId']
            session['sucess'] = True
            return redirect(url_for('atssim'))
        except :
            sucess=False
            return render_template("loginnew.html",statuslogin="user not found or password is not correct")
    return render_template("loginnew.html")

#signup
@app.route('/signup', methods=['POST','GET'])
def signup():
    if request.method == "POST":
        if 'submit' in request.form:
            usersign = request.form.get("usersign")
            passwordsign= request.form.get("passwordsign")
            conpasswordsign= request.form.get("conpasswordsign")
            if len(passwordsign) < 8:
                statussignup='Password is too short'
                return render_template('signupnew.html',statussignup=statussignup)
            try:
                usersign = auth.sign_in_with_email_and_password(usersign, passwordsign)
                statussignup='this email have been signed up'
                return render_template('signupnew.html',statussignup=statussignup)
            except:
                # User does not exist
                if passwordsign==conpasswordsign:
                    userid=auth.create_user_with_email_and_password(usersign,passwordsign)
                    db.child(userid['localId']).child("ID").set(userid['localId'])
                    db.child(userid['localId']).child("Email").set(userid)
                    db.child(userid['localId']).child("Password").set(passwordsign)
                    db.child(userid['localId']).child("MONEY").set(10000000)
                    statussignup='SUCCESS'   
                    return render_template('signupnew.html',statussignup=statussignup)
                else:
                    statussignup="Password not match"
                    return render_template('signupnew.html',statussignup=statussignup)
            # yeettopathdb(usersign,[usersign,passwordsign])
    return render_template("signupnew.html")

#research
@app.route('/atssim', methods=['POST','GET'])
def atssim():
    if sucees==False:
        return redirect(url_for('login'))
    sucess = session.get('sucess')
    if sucess==False:
        return redirect(url_for('login'))
    findindis("SCC.BK")
    sellbuy,result,purebh,ret=check("SCC")
    fig = px.line(sellbuy, x='Date', y="adj_close_price")
    fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
    fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
    fig.update_layout(title={'text': "SCC",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
    fig.update_layout(width=1200, height=500)
    graphJSON5 = plotly.io.to_json(fig)
    if ret<0:
        colorret="red"
    elif ret>0:
        colorret="green"
    else:
        colorret="grey"
    if purebh<0:
        colorbh="red"
    elif purebh>0:
        colorbh="green"
    else:
        colorbh="grey"
    if request.method == "POST":

        marketname = request.form.get("selectmarket")
        stockname = request.form.get("stockname")
        if ret<0:
            colorret="red"
        elif ret>0:
            colorret="green"
        else:
            colorret="grey"
        if purebh<0:
            colorret="red"
        elif purebh>0:
            colorbh="green"
        else:
            colorbh="grey"
        if marketname == "NONE" or stockname=="":
            return render_template("Research.html",graphJSON1=graphJSON1,graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4,graphJSON5=graphJSON5,marketname="NONE",stockname="NONE",purebh="N/A",result="N/A",ret="N/A",colorret="grey",colorbh="grey",purebh3=purebh3,purebh2=purebh2,purebh1=purebh1,purebh4=purebh4,colorbh4=colorbh4,colorbh3=colorbh3,colorbh2=colorbh2,colorret4=colorret4,colorret3=colorret3,colorret2=colorret2,colorret1=colorret1,ret1=ret1,ret2=ret2,ret3=ret3,ret4=ret4,colorbh1=colorbh1)
        else:
            if marketname=="THAI":
                tickername=str(stockname)+".BK"
                try:
                    findindis(tickername)
                    sellbuy,result,purebh,ret=check(stockname)
                    fig = px.line(sellbuy, x='Date', y="adj_close_price")
                    fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
                    fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
                    fig.update_layout(title={'text': stockname,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
                    fig.update_layout(width=1200, height=500)
                    graphJSON5 = plotly.io.to_json(fig)
                    if ret<0:
                        colorret="red"
                    elif ret>0:
                        colorret="green"
                    else:
                        colorret="grey"
                    if purebh<0:
                        colorbh="red"
                    elif purebh>0:
                        colorbh="green"
                    else:
                        colorbh="grey"
                    return render_template("Research.html",graphJSON1=graphJSON1,graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4,graphJSON5=graphJSON5,marketname=marketname,stockname=stockname,purebh=purebh,result=result,ret=ret,colorret=colorret,colorbh=colorbh,purebh3=purebh3,purebh2=purebh2,purebh1=purebh1,purebh4=purebh4,colorbh4=colorbh4,colorbh3=colorbh3,colorbh2=colorbh2,colorret4=colorret4,colorret3=colorret3,colorret2=colorret2,colorret1=colorret1,ret1=ret1,ret2=ret2,ret3=ret3,ret4=ret4,colorbh1=colorbh1)
                except:
                    stockname="NONE"
                    if ret<0:
                        colorret="red"
                    elif ret>0:
                        colorret="green"
                    else:
                        colorret="grey"
                    if purebh<0:
                        colorret="red"
                    elif purebh>0:
                        colorbh="green"
                    else:
                        colorbh="grey"
                    return render_template("Research.html",graphJSON1=graphJSON1,graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4,graphJSON5=graphJSON5,marketname=marketname,stockname=stockname,purebh=purebh,result=result,ret=ret,colorret=colorret,colorbh=colorbh,purebh3=purebh3,purebh2=purebh2,purebh1=purebh1,purebh4=purebh4,colorbh4=colorbh4,colorbh3=colorbh3,colorbh2=colorbh2,colorret4=colorret4,colorret3=colorret3,colorret2=colorret2,colorret1=colorret1,ret1=ret1,ret2=ret2,ret3=ret3,ret4=ret4,colorbh1=colorbh1)
            try:
                findindis(stockname)
                sellbuy,result,purebh,ret=check(stockname)
                fig = px.line(sellbuy, x='Date', y="adj_close_price")
                fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
                fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
                fig.update_layout(title={'text': stockname,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
                fig.update_layout(width=1200, height=500)
                graphJSON5 = plotly.io.to_json(fig)
                if ret<0:
                    colorret="red"
                elif ret>0:
                    colorret="green"
                else:
                    colorret="grey"
                if purebh<0:
                    colorbh="red"
                elif purebh>0:
                    colorbh="green"
                else:
                    colorbh="grey"
                return render_template("Research.html",graphJSON1=graphJSON1,graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4,graphJSON5=graphJSON5,marketname=marketname,stockname=stockname,purebh=purebh,result=result,ret=ret,colorret=colorret,colorbh=colorbh,purebh3=purebh3,purebh2=purebh2,purebh1=purebh1,purebh4=purebh4,colorbh4=colorbh4,colorbh3=colorbh3,colorbh2=colorbh2,colorret4=colorret4,colorret3=colorret3,colorret2=colorret2,colorret1=colorret1,ret1=ret1,ret2=ret2,ret3=ret3,ret4=ret4,colorbh1=colorbh1)
            except:
                stockname="NONE"
                if ret<0:
                    colorret="red"
                elif ret>0:
                    colorret="green"
                else:
                    colorret="grey"
                if purebh<0:
                    colorret="red"
                elif purebh>0:
                    colorbh="green"
                else:
                    colorbh="grey"
                return render_template("Research.html",graphJSON1=graphJSON1,graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4,graphJSON5=graphJSON5,marketname=marketname,stockname=stockname,purebh=purebh,result=result,ret=ret,colorret=colorret,colorbh=colorbh,purebh3=purebh3,purebh2=purebh2,purebh1=purebh1,purebh4=purebh4,colorbh4=colorbh4,colorbh3=colorbh3,colorbh2=colorbh2,colorret4=colorret4,colorret3=colorret3,colorret2=colorret2,colorret1=colorret1,ret1=ret1,ret2=ret2,ret3=ret3,ret4=ret4,colorbh1=colorbh1)
            
    return render_template("Research.html",graphJSON1=graphJSON1,graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4,graphJSON5=graphJSON5,purebh=purebh,result=result,ret=ret,colorret=colorret,colorbh=colorbh,purebh3=purebh3,purebh2=purebh2,purebh1=purebh1,purebh4=purebh4,colorbh4=colorbh4,colorbh3=colorbh3,colorbh2=colorbh2,colorret4=colorret4,colorret3=colorret3,colorret2=colorret2,colorret1=colorret1,ret1=ret1,ret2=ret2,ret3=ret3,ret4=ret4,colorbh1=colorbh1)

#livetrade
@app.route('/trade', methods=['POST','GET'])
def trade():
    findindis("AAPL")
    sellbuy6,result6,purebh6,ret6=check("AAPL")
    fig = px.line(sellbuy6, x='Date', y="adj_close_price")
    fig.update_layout(title={'text': "AAPL",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price",font=dict(size=12))
    # fig.update_layout(width=1400, height=600)
    graphJSON6 = plotly.io.to_json(fig)
    if sucees==False:
            return redirect(url_for('login'))
    sucess = session.get('sucess')
    if sucess==False:
        return redirect(url_for('login'))
    import findindi
    dfo=findindi.c

    masig=[['sma89_signal',"SMA 89","",""],['sma100_signal',"SMA 100","",""],
            ['ma5_10_signal',"MA CROSS 5/10","",""],['ma21_89_signal',"MA CROSS 21/89","",""],
            ['ma25_89_signal',"MA CROSS 25/89","",""],['ma30_50_signal',"MA CROSS 30/50","",""],['ma30_100_signal',"MA CROSS 30/100","",""],
            ['ma50_100_signal',"MA CROSS 50/100","",""],['ma30_200_signal',"MA CROSS 30/200","",""],['ma50_200_signal',"MA CROSS 50/200","",""]]
    tisig=[['psar_changetrend_signal',"PSAR","",""],['renko_signal',"RENKO","",""],
            ['macd_crossover_signal',"MACD CROSS","",""],['cdc_signal',"CDC","",""],
            ['rsi_signal',"RSI","",""],['rsi_divergent_signal',"RSI DIVERGENT","",""],['volume_prof_signal',"VOLUME PROFILE","",""],
            ['stoch_signal',"STOCHRSI","",""]]
    contisig=[['rsi_adx_signal',"RSI&ADX",""],['real_obv_macd_crossover_signal',"OBV&MACD CROSS",""],
            ['renko_macdzero_signal',"RENKO&MACD",""],['psar_adx_signal',"PSAR&ADX",""],
            ['bb_rsi_signal',"BB&RSI",""],['stoch_adx_signal',"STOCH&ADX",""]]
    volititesig=[["ADX","",""],["ATR","",""]]

    desig=["ATS  (DE)",""]
    if request.method == "POST":
        marketname = request.form.get("selectmarket")
        symbol = request.form.get("symbol")
        if symbol=="" or  symbol is None or marketname=="" or marketname is None:
            return render_template("trade.html",masig=masig,tisig=tisig,contisig=contisig,desig=desig,volititesig=volititesig,graphJSON6=graphJSON6)
        try:
            if marketname=="THAI":
                tickername=str(symbol)+".BK"

                newtechnical=findindis(tickername)
                sellbuy,result,purebh,ret=check(symbol)
                fig = px.line(sellbuy, x='Date', y="adj_close_price")
                fig.update_layout(title={'text': symbol,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price",font=dict(size=12))
                fig.update_layout(width=1400, height=600)
                graphJSON6 = plotly.io.to_json(fig)
                desig[1]=result
                ADX=round(get_adx_value()[len(get_adx_value())-1],2)
                atr=round(get_atr_value()[len(get_atr_value())-1],2)
                if ADX >= 20:
                    volititesig[0][2]="High Volatility"
                    volititesig[0][1]=ADX
                else:
                    volititesig[0][2]="Low Volatility"
                    volititesig[0][1]=ADX
                sl1=newtechnical['adx_signal']
                if atr >= 20:
                    volititesig[1][2]="High Volatility"
                    volititesig[1][1]=atr
                else:
                    volititesig[1][2]="Low Volatility"
                    volititesig[1][1]=atr
                sl1=newtechnical['adx_signal']
                masig[1][3]=round(get_ma100_value()[len(get_ma100_value())-1],2)
                masig[0][3]=round(get_ma89_value()[len(get_ma89_value())-1],2)
                tisig[4][3]=round(get_rsi_value()[len(get_rsi_value())-1],2)
                tisig[2][3]=round(get_macd_value()[len(get_macd_value())-1],2)
                tisig[3][3]=round(get_cdc_value()[len(get_cdc_value())-1],2)
                tisig[7][3]=round(get_stoch_value()[len(get_stoch_value())-1],2)
                tisig[0][3]=round(get_psar_value()[len(get_psar_value())-1],2)
                tisig[5][3]=round(get_rsi_value()[len(get_rsi_value())-1],2)
                tisig[6][3]=round(get_volume_value()[len(get_volume_value())-1],2)


                for i in masig:
                    a=0
                    j=newtechnical[i[0]][len(sl1)-1]
                    if j == -1:
                        i[2]="SELL"
                    elif j == 1:
                        i[2]="BUY"
                    else:
                        i[2]="HOLD"
                    a+=1
                for i in tisig:
                    a=0
                    j=newtechnical[i[0]][len(sl1)-1]
                    if j == -1:
                        i[2]="SELL"
                    elif j == 1:
                        i[2]="BUY"
                    else:
                        i[2]="HOLD"
                    a+=1
                for i in contisig:
                    a=0
                    j=newtechnical[i[0]][len(sl1)-1]
                    if j == -1:
                        i[2]="SELL"
                    elif j == 1:
                        i[2]="BUY"
                    else:
                        i[2]="HOLD"
                    a+=1
                return render_template("trade.html",symbol=symbol,masig=masig,tisig=tisig,contisig=contisig,desig=desig,volititesig=volititesig,graphJSON6=graphJSON6)
            else:
                newtechnical=findindis(symbol)
                sellbuy,result,purebh,ret=check(symbol)
                sellbuy,result,purebh,ret=check(symbol)
                fig = px.line(sellbuy, x='Date', y="adj_close_price")
                fig.update_layout(title={'text': symbol,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price",font=dict(size=12))
                # fig.update_layout(width=1400, height=600)
                graphJSON6 = plotly.io.to_json(fig)
                desig[1]=result
                ADX=round(get_adx_value()[len(get_adx_value())-1],2)
                atr=round(get_atr_value()[len(get_atr_value())-1],2)
                if ADX >= 20:
                    volititesig[0][2]="High Volatility"
                    volititesig[0][1]=ADX
                else:
                    volititesig[0][2]="Low Volatility"
                    volititesig[0][1]=ADX
                sl1=newtechnical['adx_signal']
                if atr >= 20:
                    volititesig[1][2]="High Volatility"
                    volititesig[1][1]=atr
                else:
                    volititesig[1][2]="Low Volatility"
                    volititesig[1][1]=atr
                sl1=newtechnical['adx_signal']
                masig[1][3]=round(get_ma100_value()[len(get_ma100_value())-1],2)
                masig[0][3]=round(get_ma89_value()[len(get_ma89_value())-1],2)
                tisig[4][3]=round(get_rsi_value()[len(get_rsi_value())-1],2)
                tisig[2][3]=round(get_macd_value()[len(get_macd_value())-1],2)
                tisig[3][3]=round(get_cdc_value()[len(get_cdc_value())-1],2)
                tisig[7][3]=round(get_stoch_value()[len(get_stoch_value())-1],2)
                tisig[0][3]=round(get_psar_value()[len(get_psar_value())-1],2)
                tisig[5][3]=round(get_rsi_value()[len(get_rsi_value())-1],2)
                tisig[6][3]=round(get_volume_value()[len(get_volume_value())-1],2)


                for i in masig:
                    a=0
                    j=newtechnical[i[0]][len(sl1)-1]
                    if j == -1:
                        i[2]="SELL"
                    elif j == 1:
                        i[2]="BUY"
                    else:
                        i[2]="HOLD"
                    a+=1
                for i in tisig:
                    a=0
                    j=newtechnical[i[0]][len(sl1)-1]
                    if j == -1:
                        i[2]="SELL"
                    elif j == 1:
                        i[2]="BUY"
                    else:
                        i[2]="HOLD"
                    a+=1
                for i in contisig:
                    a=0
                    j=newtechnical[i[0]][len(sl1)-1]
                    if j == -1:
                        i[2]="SELL"
                    elif j == 1:
                        i[2]="BUY"
                    else:
                        i[2]="HOLD"
                    a+=1
                return render_template("trade.html",symbol=symbol,masig=masig,tisig=tisig,contisig=contisig,desig=desig,volititesig=volititesig,graphJSON6=graphJSON6)
        except:
            return render_template("trade.html",symbol=symbol,masig=masig,tisig=tisig,contisig=contisig,desig=desig,volititesig=volititesig,graphJSON6=graphJSON6)

    return render_template("trade.html",masig=masig,tisig=tisig,contisig=contisig,desig=desig,volititesig=volititesig,graphJSON6=graphJSON6)



    #return render_template("trade.html",symbol="TSLA",masig=masig,tisig=tisig,contisig=contisig,desig=desig,volititesig=volititesig,graphJSON6=graphJSON6)sig,contisig=contisig,desig=desig,volititesig=volititesig,graphJSON6=graphJSON6)

#portofoliio
@app.route('/tradee', methods=['POST','GET'])
def tradee():
    if sucees==False:
        return redirect(url_for('login'))
    sucess = session.get('sucess')
    if sucess==False:
        return redirect(url_for('login'))
    ID = session.get('ID')
    buyorder=db.child(ID).child("BUY").get().val()
    sellorder=db.child(ID).child("SELL").get().val()
    ava=db.child(ID).child("MONEY").get().val()
    buylist=db.child(ID).child("BUY").get().val()
    selllist=db.child(ID).child("SELL").get().val()
    if buylist is not None or selllist is not None :
        if buylist is not None and selllist is  None :
            thia={}
            for o in buylist:
                sum=0
                for i in count_stock_in_port(buylist)[o[0]]:
                    sum+=int(i)
            thia[o[0]]=sum
            thib={}
            d1 = Counter(thia)
            d2 = Counter(thib)
            d3 = d1 - d2
            inputDictionary = d3
            resultList = list(inputDictionary.keys())
            resultLista = list(inputDictionary.values())
            length=len(resultList)
        if buylist is  None and selllist is not  None :
            thia={}
            thib={}
            for o in selllist:
                sum=0
                for i in count_stock_in_port(selllist)[o[0]]:
                    sum+=int(i)
            thib[o[0]]=sum
            d1 = Counter(thia)
            d2 = Counter(thib)
            d3 = d1 - d2
            inputDictionary = d3
            resultList = list(inputDictionary.keys())
            resultLista = list(inputDictionary.values())
            length=0
        if buylist is not None and selllist is not  None :
            thia={}
            for o in buylist:
                sum=0
                for i in count_stock_in_port(buylist)[o[0]]:
                    sum+=int(i)
            thia[o[0]]=sum
            thib={}
            for o in selllist:
                sum=0
                for i in count_stock_in_port(selllist)[o[0]]:
                    sum+=int(i)
            thib[o[0]]=sum
            d1 = Counter(thia)
            d2 = Counter(thib)
            d3 = d1 - d2
            inputDictionary = d3
            resultList = list(inputDictionary.keys())
            resultLista = list(inputDictionary.values())
            length=len(resultList)
    else:
        resultList = ""
        resultLista = ""
        length=0
    # printing the resultant list of a dictionary keys
    if buyorder is None:
        buyorder=[["None","None","None"]]   
    if sellorder is None:
        sellorder=[["None","None","None"]]
    show_message=False
    if 'confirm' in request.form:
        namebuy = session.get('namebuy')
        price = session.get('price')
        amount = session.get('amount')
        selectmethod = session.get('selectmethod')
        cost = session.get('cost')
        ID = session.get('ID')
        if selectmethod=="BUY":
            ava=db.child(ID).child("MONEY").get().val()
            if ava-cost<0:
                return render_template("trade2.html",buyorder=buyorder,sellorder=sellorder)
            db.child(ID).child("MONEY").set(ava-cost)
            if db.child(ID).child(selectmethod).get().val() is None:
                db.child(ID).child(selectmethod).child(0).set([namebuy,price,amount])
            else:
                numbertrans=len(db.child(ID).child(selectmethod).get().val())
                db.child(ID).child(selectmethod).child(numbertrans).set([namebuy,price,amount])
                buyorder=db.child(ID).child("BUY").get().val()
                sellorder=db.child(ID).child("SELL").get().val()
            return render_template("trade2.html",amount=amount,namebuy=namebuy,selectmethod=selectmethod,price=price,cost=cost,show_message=False,buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista,length=length)
        if selectmethod=="SELL":
            ava=db.child(ID).child("MONEY").get().val()
            buylist=db.child(ID).child("BUY").get().val()
            selllist=db.child(ID).child("SELL").get().val()
            buydict={}
            for o in buylist:
                sum=0
                for i in count_stock_in_port(buylist)[o[0]]:
                    sum+=int(i)
                buydict[o[0]]=sum
            selldict={}
            try:
                for o in selllist:
                    sum=0
                    for i in count_stock_in_port(selllist)[o[0]]:
                        sum+=int(i)
                    selldict[o[0]]=sum
            except:
                pass
            if selldict.get(namebuy) is None:
                if amount > buydict.get(namebuy):
                    return render_template("trade2.html",buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista)
            if selldict.get(namebuy) is not None:
                if buydict.get(namebuy) is None :
                    return render_template("trade2.html",buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista)
               
                if amount > buydict.get(namebuy)-selldict.get(namebuy):
                    return render_template("trade2.html",buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista)
            if buydict.get(namebuy) is None:
                return render_template("trade2.html",buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista)
            if amount > buydict.get(namebuy):
                return render_template("trade2.html",buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista)
            db.child(ID).child("MONEY").set(ava+cost)
            if db.child(ID).child(selectmethod).get().val() is None:
                db.child(ID).child(selectmethod).child(0).set([namebuy,price,amount])
                
                return render_template("trade2.html",amount=amount,namebuy=namebuy,selectmethod=selectmethod,price=price,cost=cost,show_message=False,buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista,length=length)
                
            else:
                numbertrans=len(db.child(ID).child(selectmethod).get().val())
                db.child(ID).child(selectmethod).child(numbertrans).set([namebuy,price,amount])
                buyorder=db.child(ID).child("BUY").get().val()
                sellorder=db.child(ID).child("SELL").get().val()
                return render_template("trade2.html",amount=amount,namebuy=namebuy,selectmethod=selectmethod,price=price,cost=cost,show_message=False,buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista,length=length)
            return render_template("trade2.html",amount=amount,namebuy=namebuy,selectmethod=selectmethod,price=price,cost=cost,show_message=False,buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista,length=length)
        return render_template("trade2.html",amount=amount,namebuy=namebuy,selectmethod=selectmethod,price=price,cost=cost,show_message=False,buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista,length=length)
    
    
    if request.method == "POST":
        if 'order' in request.form:
            try:
                selectmethod = request.form.get("selectmethod")
                selectmarket = request.form.get("selectmarket")
                namebuy = request.form.get("namebuy").upper()
                amount= request.form.get("amount")
                if amount is not None  and namebuy is not None  and selectmethod != "NONE" and selectmarket != "NONE":
                    amount=int(amount)
                    if selectmarket =="THAI":
                        stocktobuy=findindis(namebuy+".BK")
                        price=round(float(stocktobuy['adj_close_price'][len(stocktobuy['adj_close_price'])-1]),2)
                    if selectmarket =="US":
                        stocktobuy=findindis(namebuy)
                        price=round(float(stocktobuy['adj_close_price'][len(stocktobuy['adj_close_price'])-1])*34,2)
                    cost=round(int(amount)*price,2)
                    session['namebuy'] = namebuy
                    session['price'] = price
                    session['amount'] = amount
                    session['selectmethod'] = selectmethod
                    session['selectmarket'] = selectmarket
                    session['namebuy'] = namebuy
                    session['cost'] = cost
                    return render_template("trade2.html",amount=amount,namebuy=namebuy,selectmethod=selectmethod,price=price,cost=cost,show_message=True,buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista,length=length)
            except:
                return render_template("trade2.html",buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista,length=length)
        return render_template("trade2.html",buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista,length=length)
    
        
        
    return render_template("trade2.html",buyorder=buyorder,sellorder=sellorder,ava=ava,resultList=resultList,resultLista=resultLista,length=length)
if __name__ == "__main__":
    app.run(debug=True)
