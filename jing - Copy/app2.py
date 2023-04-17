# from flask import Flask,render_template,url_for,request
# from findindi import *
# from flask_sqlalchemy import SQLAlchemy
# from checksignal import *
# import pandas as pd
# from yeet import*
# import json
# import plotly
# import plotly.express as px
# import plotly.graph_objects as go
# app= Flask(__name__)
# # app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///data.db'
# setupdb()
# status=False
# graphexwid=350
# graphexheight=300



# findindis("CPALL.BK")
# sellbuy,b=check()
# fig = px.line(sellbuy, x='Date', y="adj_close_price")
# fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
# fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
# fig.update_layout(title={'text': "CPALL.BK",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
# fig.update_layout(width=graphexwid, height=graphexheight)
# graphJSON1 = plotly.io.to_json(fig)

# findindis("ADVANC.BK")
# sellbuy,b=check()
# fig = px.line(sellbuy, x='Date', y="adj_close_price")
# fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
# fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
# fig.update_layout(title={'text': "ADVANC.BK",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
# fig.update_layout(width=graphexwid, height=graphexheight)
# graphJSON2 = plotly.io.to_json(fig)

# findindis("AOT.BK")
# sellbuy,b=check()
# fig = px.line(sellbuy, x='Date', y="adj_close_price")
# fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
# fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
# fig.update_layout(title={'text': "AOT.BK",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
# fig.update_layout(width=graphexwid, height=graphexheight)
# graphJSON3 = plotly.io.to_json(fig)

# findindis("PTT.BK")
# sellbuy,b=check()
# fig = px.line(sellbuy, x='Date', y="adj_close_price")
# fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
# fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
# fig.update_layout(title={'text': "PTT.BK",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
# fig.update_layout(width=graphexwid, height=graphexheight)
# graphJSON4 = plotly.io.to_json(fig)
# @app.route('/', methods=['POST','GET'])
# def home():
#     findindis("OR.BK")
#     sellbuy,b=check()
#     fig = px.line(sellbuy, x='Date', y="adj_close_price")
#     fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
#     fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
#     fig.update_layout(title={'text': "OR.BK",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
#     fig.update_layout(width=1500, height=500)
#     graphJSON5 = plotly.io.to_json(fig)
#     if request.method == "POST":
#         stockname = request.form.get("stockname")
#         if stockname == "":
#             return render_template("index2.html",graphJSON1=graphJSON1,graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4)
#         else:
#             try:
#                 findindis(stockname)
#                 sellbuy,b=check()
#                 fig = px.line(sellbuy, x='Date', y="adj_close_price")
#                 fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
#                 fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
#                 fig.update_layout(title={'text': stockname,'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
#                 fig.update_layout(width=1500, height=500)
#                 graphJSON5 = plotly.io.to_json(fig)
#                 return render_template("index2.html",graphJSON1=graphJSON1,graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4,graphJSON5=graphJSON5)
#             except:
#                 findindis("OR.BK")
#                 sellbuy,b=check()
#                 fig = px.line(sellbuy, x='Date', y="adj_close_price")
#                 fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['buy'], mode='markers', name="BUY", marker=dict(color='Green', size=12)))
#                 fig.add_trace(go.Scatter(x=sellbuy['Date'], y=sellbuy['sell'], mode='markers', name="SELL", marker=dict(color='Red', size=12)))
#                 fig.update_layout(title={'text': "OR.BK",'x':0.5,'xanchor': 'center','yanchor': 'top'},xaxis_title="Date",yaxis_title="Price in BAHT",font=dict(size=12))
#                 fig.update_layout(width=1500, height=500)
#                 graphJSON5 = plotly.io.to_json(fig)
#                 return render_template("index2.html",graphJSON1=graphJSON1,graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4,graphJSON5=graphJSON5)
    
    

#     return render_template("index2.html",graphJSON1=graphJSON1,graphJSON2=graphJSON2,graphJSON3=graphJSON3,graphJSON4=graphJSON4,graphJSON5=graphJSON5)

# @app.route('/about', methods=['POST','GET'])
# def about():
#     if request.method == "POST":
#         if 'submit' in request.form:
#             usersign = request.form.get("usersign")
#             passwordsign= request.form.get("passwordsign")
#             yeettopathdb(usersign,[usersign,passwordsign])
#             statussignup='SUCESS'
#             return render_template('signup.html')
#     return render_template('signup.html')
# statuslogin=''
# @app.route('/contact', methods=['POST','GET'])
# def contact():
#     if request.method == "POST":
#         if 'submit' in request.form:
#             user = request.form.get("user")
#             password= request.form.get("password")
#             checkuser=getdatadb(user)
#             if checkuser[0]==user and checkuser[1]==password:
#                 statuslogin='SUCESS'

#             return render_template('login.html',statuslogin=statuslogin)
#     return render_template('login.html')

# if __name__ == "__main__":
#     app.run(debug=True)