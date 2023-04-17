#cd Desktop\program ana\new\Streamlit-20230222T143615Z-001\Streamlit Streamlit run streamlit1.py
import streamlit as st
from findindi import*
import yfinance as yf
from yeet import*
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
t=True
box_style = """
<style>
    # #MainMenu {visibility: hidden; }
    # footer {visibility: hidden;}

    body{
    background-color: #f1f1f1; 
    }
    .wrapper {
        border: 1px solid #ccc; /* add a 1px gray border to the wrapper */
        padding: 20px; /* add some padding to the wrapper to create space around the content */
        }
    .box {
        background-color:  #282434;
        color: white;
        border: 1px solid}
    .headername{
        font-size: 36px;
        font-weight: bolder;
        margin-left: 10px;
        color: black;
        padding-top: 0px;
        text-align: left;
        }
    hr {
        height: 5px;
        margin-top:0px;
        background-color: black;
        }
    .myblock {
        background-color: #F0F0F0;
        color: white;
        border: 1px solid #C0C0C0;
        }
    .bigfont{
        font-size: 20px;
        font-weight: bold;
        color: black;
        
        text-align: center;
        width: fit-content}
    [class="css-ocqkz7 e1tzin5v4"]  {
        border: 1px solid black;
        padding: 10px;
    }
    .font-green{
        color:green;
        font-weight: bold;
        margin-top: -10px;
    }
    .font-red{
        color:red;
        font-weight: bold;
        margin-top: -10px;
    }
    .font-grey{
        color:grey;
        font-weight: bold;
        margin-top: -10px;
    }
    .justbold{
        color: black;
        font-weight: bold;
       
        margin-bottom: -700px;
    }
</style>
"""
st.set_page_config(page_title="My Streamlit App", page_icon=":guardsman:", layout="wide")

def login(username, password):
    if username == "user" and password == "password":
        return True
    else:
        return False



def check():
    import findindi
    import pandas as pd
    import numpy as np
    global signal,buy,sell,dfo,sellbuy,ret,purebh
    amount,ret=0,0
    dfo=findindi.c
    sl1=dfo['adx_signal']
    sl2=dfo['macd_signal']
    sl3=dfo['ma30_200_signal']
    sl4=dfo['ma50_200_signal']
    sl5=dfo['ma30_50_signal']
    sl6=dfo['cdc_signal']
    sl7=dfo['sar_signal']
    sl8=dfo['roc_signal']
    sl9=dfo['obv_signal']
    sl10=dfo['aroon_signal']
    sl11=dfo['adj_close_price']
    w=[]
    # for i in range(10):
    #     w.append(0.1)
    w=[0.10607762413686823, 0.15156331863166905, 0.07883386165874004,
    0.06871414126119149,  0.0852604234155046, 0.13452143197313385,
    0.12098422686625537, 0.06112249091135928, 0.06255874149627269,
    0.13036373964900538]
    td=0.2
    mon=100000
    buy=[]
    sell=[]
    signala=0
    signal = []
    stockbuy=mon//sl11[0]
    purebh=(sl11[-1]*stockbuy)
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
    ret=mon+amount*dfo['adj_close_price'][len(dfo)-1]
    rai=dfo.drop(columns=['close_price','macd_signal','aroon_signal','obv_signal','roc_signal','sar_signal','cdc_signal','ma30_50_signal','ma50_200_signal','ma30_200_signal','adx_signal'])
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
    return sellbuy,result
def test():
    fig = px.line(sellbuy, x='Date', y="adj_close_price")
    fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict( color='Green', size=8,)))
    fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy["sell"], mode='markers', name="SELL", marker=dict( color='Red', size=8,)))
    fig.update_layout(title={'text': "CPALL",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in USD",width=300, height=300 ,font=dict(size=20))
    st.plotly_chart(fig, theme="streamlit", use_container_width=False, width=300, height=300,font=dict(size=20),line=dict(width=3))

st.markdown(box_style, unsafe_allow_html=True)
st.markdown("<div class='headername'><b>Algorithmic Trading</b></div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("<p class='bigfont'>EXAMPLE</p>", unsafe_allow_html=True)
container = st.container()
container1 = st.container()
box_container = st.container()

@st.cache_data
def example():
    
    with container:
        columns = st.columns(4)
    
        with columns[0]:
            findindis("AAPL")
            check()
            fig = px.line(sellbuy, x='Date', y="adj_close_price")
            fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict( color='Green', size=6,)))
            fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy["sell"], mode='markers', name="SELL", marker=dict( color='Red', size=6,)))
            fig.update_layout(title={'text': "APPLE",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in USD",width=300, height=300 ,font=dict(size=20))
            st.plotly_chart(fig, theme="streamlit", use_container_width=False, width=300, height=300,font=dict(size=20),line=dict(width=3))
            if  result == "SELL":
                st.markdown(f"<p class='font-green'>Recommendation: {result}</div>", unsafe_allow_html=True)
            elif result == "BUY":
                st.markdown(f"<p class='font-red'>Recommendation: {result}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-grey'>Recommendation: {result}</div>", unsafe_allow_html=True)
            if (ret-100000)/1000 > 0:
                st.markdown(f"<p class='font-green'>Your Money: {(ret-100000)/1000} %</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-red'>Your Money: {(ret-100000)/1000} %</div>", unsafe_allow_html=True)
            if (purebh-100000)/1000 > 0:
                st.markdown(f"<p class='font-green'>B&H: {(purebh-100000)/1000} %", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-red'>B&H: {(purebh-100000)/1000} %", unsafe_allow_html=True) 

        with columns[1]:
            findindis("TSLA")
            check()
            fig = px.line(sellbuy, x='Date', y="adj_close_price")
            fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict( color='Green', size=8,)))
            fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy["sell"], mode='markers', name="SELL", marker=dict( color='Red', size=8,)))
            fig.update_layout(title={'text': "TESLA",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in USD",width=300, height=300 ,font=dict(size=20))
            st.plotly_chart(fig, theme="streamlit", use_container_width=False, width=300, height=300,font=dict(size=20),line=dict(width=3))
            if  result == "SELL":
                st.markdown(f"<p class='font-green'>Recommendation: {result}</div>", unsafe_allow_html=True)
            elif result == "BUY":
                st.markdown(f"<p class='font-red'>Recommendation: {result}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-grey'>Recommendation: {result}</div>", unsafe_allow_html=True)
            if (ret-100000)/1000 > 0:
                st.markdown(f"<p class='font-green'>Your Money: {(ret-100000)/1000} %</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-red'>Your Money: {(ret-100000)/1000} %</div>", unsafe_allow_html=True)
            if (purebh-100000)/1000 > 0:
                st.markdown(f"<p class='font-green'>B&H: {(purebh-100000)/1000} %", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-red'>B&H: {(purebh-100000)/1000} %", unsafe_allow_html=True) 
        with columns[2]:
            findindis("BTS.BK")
            check()
            fig = px.line(sellbuy, x='Date', y="adj_close_price")
            fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict( color='Green', size=8,)))
            fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy["sell"], mode='markers', name="SELL", marker=dict( color='Red', size=8,)))
            fig.update_layout(title={'text': "BTS",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in USD",width=300, height=300 ,font=dict(size=20))
            st.plotly_chart(fig, theme="streamlit", use_container_width=False, width=300, height=300,font=dict(size=20),line=dict(width=3))
            if  result == "SELL":
                st.markdown(f"<p class='font-green'>Recommendation: {result}</div>", unsafe_allow_html=True)
            elif result == "BUY":
                st.markdown(f"<p class='font-red'>Recommendation: {result}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-grey'>Recommendation: {result}</div>", unsafe_allow_html=True)
            if (ret-100000)/1000 > 0:
                st.markdown(f"<p class='font-green'>Your Money: {(ret-100000)/1000} %</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-red'>Your Money: {(ret-100000)/1000} %</div>", unsafe_allow_html=True)
            if (purebh-100000)/1000 > 0:
                st.markdown(f"<p class='font-green'>B&H: {(purebh-100000)/1000} %", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-red'>B&H: {(purebh-100000)/1000} %", unsafe_allow_html=True) 
        with columns[3]:
            findindis("CPALL.BK")
            check()
            fig = px.line(sellbuy, x='Date', y="adj_close_price")
            fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict( color='Green', size=8,)))
            fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy["sell"], mode='markers', name="SELL", marker=dict( color='Red', size=8,)))
            fig.update_layout(title={'text': "CPALL",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in USD",width=300, height=300 ,font=dict(size=20))
            st.plotly_chart(fig, theme="streamlit", use_container_width=False, width=300, height=300,font=dict(size=20),line=dict(width=3))
            if  result == "SELL":
                st.markdown(f"<p class='font-green'>Recommendation: {result}</div>", unsafe_allow_html=True)
            elif result == "BUY":
                st.markdown(f"<p class='font-red'>Recommendation: {result}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-grey'>Recommendation: {result}</div>", unsafe_allow_html=True)
            if (ret-100000)/1000 > 0:
                st.markdown(f"<p class='font-green'>Your Money: {(ret-100000)/1000} %</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-red'>Your Money: {(ret-100000)/1000} %</div>", unsafe_allow_html=True)
            if (purebh-100000)/1000 > 0:
                st.markdown(f"<p class='font-green'>B&H: {(purebh-100000)/1000} %", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-red'>B&H: {(purebh-100000)/1000} %", unsafe_allow_html=True)    

@st.cache_resource
def connection():
    setupdb()
loginsucess=False

def main():
    with st.sidebar:
        st.title("User Management")
        optionsuser = ["Sign Up","login"]
        loginoptions = st.selectbox(" ",optionsuser)
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button(f"{loginoptions}"):
            loginbutt=True
        else:
            loginbutt=False
        if loginoptions =="Sign Up"and loginbutt==True:
            yeettopathdb(username,[username,password])
            st.markdown("User Created")
        if loginoptions=="login"and loginbutt==True:
            checkuser= getdatadb(username)
            if username == checkuser[0] and password ==checkuser[1]:
                st.markdown("Login Sucess")
                loginsucess=True
            else:
                st.markdown("Login Failed")
    left_column, right_column = st.columns([2, 1])
    con=connection()
    cdw=example()
    st.markdown("<p class='bigfont'>OUR ALGO-TRADING</p>", unsafe_allow_html=True)

    columns1 = st.columns(2)

    # Add content to each column
    
    with columns1[0]:
        options = ["NONE","US", "THAI"]
        st.markdown("<p class='justbold'>Select options:</p>",unsafe_allow_html=True)
        selected_options = st.selectbox(" ",options)
        # Add content to the second column
    with columns1[1]:
        st.markdown("<p class='justbold'>Enter stock name:</p>",unsafe_allow_html=True)
        stockname = st.text_input(" ")
    with columns1[0]:
        if stockname == "" or selected_options == "NONE":
            st.write("You selected: NONE")
        else:
            # company = yf.Ticker(stockname)
            # companyname=company.info['longName']
            st.write("You selected:", stockname ," in the", selected_options ," market")
        if selected_options=="THAI":
            tickername=str(stockname)+".BK"
        else:
            tickername=str(stockname)
        if st.button("Confirm"):
            checkbutt=True
        else:
            checkbutt=False
    if checkbutt==True:
        try:
            # st.markdown(box_style, unsafe_allow_html=True)
            # st.markdown(f"<div class='box'><p><b>RECOMMENDED : {result}</b></p><p>Your Money: {ret}</p><p>B&H : {purebh}</p></div>", unsafe_allow_html=True)

            
            findindis(tickername)
            check()
            fig = px.line(sellbuy, x='Date', y="adj_close_price")
            fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict( color='Green', size=12,)))
            fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy["sell"], mode='markers', name="SELL", marker=dict( color='Red', size=12,)))
            fig.update_layout(xaxis_title="DATE",yaxis_title="Stock Price")
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)

            if  result == "SELL":
                st.markdown(f"<p class='font-green'>Recommendation: {result}</div>", unsafe_allow_html=True)            
            elif result == "BUY":
                st.markdown(f"<p class='font-red'>Recommendation: {result}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-grey'>Recommendation: {result}</div>", unsafe_allow_html=True)
            if (ret-100000)/1000 > 0:
                st.markdown(f"<p class='font-green'>Your Money: {(ret-100000)/1000} %</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-red'>Your Money: {(ret-100000)/1000} %</div>", unsafe_allow_html=True)
            if (purebh-100000)/1000 > 0:
                st.markdown(f"<p class='font-green'>B&H: {(purebh-100000)/1000} %", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='font-red'>B&H: {(purebh-100000)/1000} %", unsafe_allow_html=True)    
            st.markdown("---")  # Add a horizontal rule

        except:
            st.error("An error occurred. Please try again later.")

        #st.markdown(f'<div class="myblock">Your Money: {ret} </div>', unsafe_allow_html=True)
    findindis("CPALL.BK")
    st.markdown(f"<p class='text-indi'>Single Individual Signal</p>",unsafe_allow_html=True)
if __name__ == "__main__":
    main()


# fig = px.line(sellbuy, x='Date', y="adj_close_price")
# fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict( color='Green', size=12,)))
# fig.add_trace(go.Scatter( x= sellbuy["Date"], y=sellbuy["sell"], mode='markers', name="SELL", marker=dict( color='Red', size=12,)))
# fig.update_layout(xaxis_title="X Axis Title",yaxis_title="Y Axis Title",width=500, height=300 ,font=dict(size=20))
# st.plotly_chart(fig, theme="streamlit", use_container_width=False, width=500, height=300,font=dict(size=20))