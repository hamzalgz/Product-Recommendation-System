#library
import os
import pathlib
import numpy as np
from dash.dependencies import Input, Output, State

import plotly.graph_objs as go
import dash_daq as daq

from dash_bootstrap_components._components.Col import Col
from dash_bootstrap_components._components.Row import Row
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_extensions import Lottie 
import plotly.express as px
import plotly.figure_factory as ff
from wordcloud import WordCloud 
import dash_bootstrap_components as dbc
import dash_table


ratingss=pd.read_csv('D:/retour/recommendation/ratingg.csv',index_col=0)
products=pd.read_csv('D:/retour/recommendation/products1.csv')
user=pd.read_csv('D:/retour/recommendation/user1.csv')
options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))
#lutties
url_11='https://assets6.lottiefiles.com/private_files/lf30_2c7wnifx.json'
url_22='https://assets5.lottiefiles.com/packages/lf20_ur7sluxh.json'
url_33='https://assets2.lottiefiles.com/packages/lf20_Y9BNoF.json'
url_4=''
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SKETCHY],
)



app.layout = dbc.Container([
    
    dbc.Row([
        dbc.Col([
            
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_33)),
                dbc.CardBody([
                    html.H6('User'),
                    
                    
                    dcc.Dropdown(
                        id='user_i',
                        options=[{'label': x, 'value': x}
                            for x in user['user_id'].unique()],     
                        value=5,

                        style={'width' : '100%',
                            'text-align': 'center',
                            }
                        
                    ),
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_22)),
                dbc.CardBody([
                    html.H6('Ratings'),
                    html.H2(id='help')
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_11)),
                dbc.CardBody([
                    html.H6('Helpings'),
                    html.H2(id='helped')
                ], style={'textAlign': 'center'})
            ]),
        ], width=2),
        
    ],className='mb-2'),
    dbc.Row([
        dbc.Col([
            
        ], width=1),
        dbc.Col([
            dbc.Card([
               
                dbc.CardBody([
                    html.H3(""),
                    
                    html.Div(id='none1',children=[]),
                    dcc.Graph(id="barr"  ,figure={}),
                ])
            ]),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dcc.Dropdown(
                        id='len',
                        options=[{'label': 'Female', 'value': 'F'},
                                {'label': 'Male', 'value': 'M'},
                            ],
                        value='F',

                        style={'width' : '70%',
                            'text-align': 'center',
                            }
                        
                    ),
                dbc.CardBody([
                    html.H3("Age distribution"),
                    dcc.Graph(id="pie-chart"  ,figure={}),
                ])
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                
                dbc.CardBody([
                    html.H3("Ratings trend"),
                    html.Div(id='none',children=[]),
                    dcc.Graph(id="pie_bar"  ,figure={}),
                ])
            ]),
        ], width=3),
    
    ],className='mb-2'),
    

    dbc.Row([
        dbc.Col([
            
        ], width=1),
        dbc.Col([
            dbc.Card([
               
                dbc.CardBody([
                    html.H3(""),
                    
                    html.Div(id='none4',children=[]),
                    dcc.Graph(id="barer"  ,figure={}),
                ])
            ]),
        ], width=5),
        
        dbc.Col([
            dbc.Card([
                
                dbc.CardBody([
                    html.H3("Ratings trend"),
                    html.Div(id='none3',children=[]),
                    dcc.Graph(id="pie-chart1"  ,figure={}),
                ])
            ]),
        ], width=3),
    
    ],className='mb-2'),

    dbc.Row([
        dbc.Col([
            
        ], width=1),
        dbc.Col([
            dbc.Card([
               
                dbc.CardBody([
                    html.H3("Rated products"),
                    html.Div(id='none2',children=[]),
                    dcc.Graph(id="histograme"  ,figure={}),
                ])
            ]),
        ], width=4),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3("Localisation"),
                    
                    dcc.Graph(id="graph")
                    
                ])
            ]),
        ], width=6),
        
    ],className='mb-2'),
], fluid=True)


#---------------
@app.callback(
    Output(component_id='help',component_property='children'),
    [Input('user_i','value')])
def helper(tweet):
    return(len(ratingss.loc[ratingss.user_id==tweet]))

@app.callback(
    Output(component_id='helped',component_property='children'),
    [Input('user_i','value')])

def helping(tweet):
    rating=ratingss.copy()

    ratings=rating[rating['user_id']==tweet]

    m=[]

    for i in ratings.product_id.tolist():
        productt=rating[rating.product_id==i]
        for j in productt.user_id.tolist():        
            m.append(j)
    x = np.array(m)
    list=np.unique(x)
    
    return(len(list))
 #Histogramm1-----------------------------------------------   


#pie-------------------------
@app.callback(
    Output(component_id='pie-chart',component_property='figure'),
    [Input('len','value')])

def update_pie(tweet):
      
    dff=user.loc[user.sex==tweet]
    fig_pie =px.pie(
        dff,
        values='age', 
        
        names=dff.age,
        hole=.5,
        color_discrete_sequence=['maroon','red','coral']
        )
    return(fig_pie)


#-------

@app.callback(
    Output(component_id='barr',component_property='figure'),
    [Input('user_i','value')])

def barr(tweet):    
    
    ratings=ratingss.copy()

    ratings=ratings[ratings['user_id']==tweet]
    ratings['Score']=ratings['rating']*20
    ratings['Count'] = ratings.groupby('rating')['rating'].transform(len)
    ratin=ratings[['Score','Count']].drop_duplicates()
    

    fig = px.bar(ratin,
                 x='Score',
                 y='Count',
                 color='Count'
                 )
    return(fig)


#---------------
@app.callback(
    Output(component_id='barer',component_property='figure'),
    [Input('user_i','value')])

def barer(tweet):    
    
    rating=ratingss.copy()

    ratings=rating[rating['user_id']==tweet]

    m=[]

    for i in ratings.product_id.tolist():
        productt=rating[rating.product_id==i]
        for j in productt.user_id.tolist():        
            m.append(j)
    x = np.array(m)
    list=np.unique(x)
    rat=[]
    for i in list:
        for j in rating[rating['user_id']==i].rating.tolist():
            rat.append(j)

    dic={'rating':rat}

    df=pd.DataFrame(dic)
    df['Score']=df['rating']*20
    df['Count'] = df.groupby('rating')['rating'].transform(len)
    df=df[['Score','Count']].drop_duplicates()

    

    fig = px.bar(df,
                 x='Score',
                 y='Count',
                 color='Count'
                 )
    return(fig)
#----------------
@app.callback(
    Output(component_id='pie_bar',component_property='figure'),
    [Input('user_i','value')])

def update_pie(tweet):

    ratings=ratingss.copy()

    ratings=ratings[ratings['user_id']==tweet]
    ratings['pourcen']=ratings['rating']*20
    ratings['count'] = ratings.groupby('rating')['rating'].transform(len)
    ratin=ratings[['pourcen','count']].drop_duplicates()
    fig_pie =px.pie(
            ratin,
            values='count', 
            
            names=ratin.pourcen,
            hole=.5,
            color_discrete_sequence=['deeppink','darkviolet']
            )
    return(fig_pie)
#---------------------------------
@app.callback(
    Output(component_id='pie-chart1',component_property='figure'),
    [Input('user_i','value')])

def update_pie(tweet):

    rating=ratingss.copy()

    ratings=rating[rating['user_id']==tweet]

    m=[]

    for i in ratings.product_id.tolist():
        productt=rating[rating.product_id==i]
        for j in productt.user_id.tolist():        
            m.append(j)
    x = np.array(m)
    list=np.unique(x)
    rat=[]
    for i in list:
        for j in rating[rating['user_id']==i].rating.tolist():
            rat.append(j)

    dic={'rating':rat}

    df=pd.DataFrame(dic)
    df['pourcen']=df['rating']*20
    df['count'] = df.groupby('rating')['rating'].transform(len)
    df=df[['pourcen','count']].drop_duplicates()
    fig_pie =px.pie(
            df,
            values='count', 
            
            names=df.pourcen,
            hole=.5,
            color_discrete_sequence=['deeppink','darkviolet']
            )
    return(fig_pie)

@app.callback(
    Output(component_id='histograme',component_property='figure'),
    [Input('user_i','value')])
def update_pie(tweet):

    ratings=ratingss.copy()
    ratings['ratings_pourcentage'] = len(ratings.groupby('user_id')['rating']) 
    ratings=ratings[ratings['user_id']==tweet]
    new=pd.merge(ratings,products,left_on='product_id',right_on='ID_product')
    new['Category']=new['category']
    fig=px.histogram(new, x="Category",color_discrete_sequence=['darkviolet'])

    return(fig)

@app.callback(
    Output("graph", "figure"), 
    [Input('user_i','value')])

def map(tweet):
    rating=ratingss.copy()

    ratings=rating[rating['user_id']==5]

    m=[]

    for i in ratings.product_id.tolist():
        productt=rating[rating.product_id==i]
        for j in productt.user_id.tolist():        
            m.append(j)
    x = np.array(m)
    list=np.unique(x)
    userr=user.copy()
    l=[]
    for i in list:
        resid=userr[userr.user_id==i]
        l.append(resid.Residence) 
    y = np.array(l)
    li=np.unique(y)
    isoo=[]
    for i in li:
        iso=np.unique(user[user.Residence==i].iso_id)
        for j in iso:
            isoo.append(j)
    residence=li.tolist()
    s={'residence':residence,'iso_id':isoo}
    df=pd.DataFrame(s)
    df['count']=df.groupby('iso_id')['iso_id'].transform(len)
  
    fig = px.scatter_geo(df, locations="iso_id",hover_name="residence",size='count',
                     projection="natural earth")
    return(fig)

# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)