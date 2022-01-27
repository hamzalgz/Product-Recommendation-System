#library
import os
import pathlib
from dash_bootstrap_components._components.Card import Card
import numpy as np
from math import pow, sqrt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from dash.dependencies import Input, Output, State
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import dash_daq as daq
import plotly.graph_objects as go
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
time=pd.read_csv('D:/retour/recommendation/time_serie.csv')

options = dict(loop=True, autoplay=True, rendererSettings=dict(preserveAspectRatio='xMidYMid slice'))
#lutties
url_1='https://assets1.lottiefiles.com/packages/lf20_xtzoykx4.json'
url_2='https://assets1.lottiefiles.com/packages/lf20_udjnduhz.json'
url_3='https://assets8.lottiefiles.com/packages/lf20_rrw4rw07.json'



app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SKETCHY],
)



app.layout = dbc.Container([
    
    dbc.Row([
        dbc.Col([
            
        ], width=1),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_3)),
                dbc.CardBody([
                    html.H6('Brand'),
                    dcc.Dropdown(
                        id='brand_id',
                        options=[{'label': x, 'value': x}
                            for x in products['brand'].unique()],     
                        value='Adidas',

                        style={'width' : '100%',
                            'text-align': 'center',
                            }
                        
                    ),
                ], style={'textAlign':'center'})
            ]),
        ], width=2),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_1)),
                dbc.CardBody([
                    html.H6('Products'),
                    html.H2(id='content-connections1')
                ], style={'textAlign': 'center'})
            ]),
        ], width=2),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_2)),
                dbc.CardBody([
                    html.H6('Ratings'),
                    html.H2(id='content-connections2')
                ], style={'textAlign':'center'})
            ]),
        ], width=2),

        dbc.Card([
            html.H3('Top 5 pickers'),
                dcc.Dropdown(
                    id='user_id',
                    options=[{'label': x, 'value': x}
                        for x in user['user_id'].unique()],     
                    value=5,

                    style={'width' : '100%',
                        'text-align': 'center',
                        }
                    
                ),
            ]),
                    
                
            dbc.Card([]),

            dbc.Card([
            html.H3('Select a product'),
                dcc.Dropdown(
                    id='name_id',
                    options=[{'label': x, 'value': x}
                        for x in products['name'].unique()],     
                    value='Sport Inspired Duramo Lite 2.0 Shoes',

                    style={'width' : '100%',
                        'text-align': 'center',
                        }
                    
                ),
            ]),
        
        
    ],className='mb-2'),
    dbc.Row([
        dbc.Col([
            
        ], width=1),
        
        dbc.Col([
            dbc.Card([
                    html.H3('Most popular products'),
                    dcc.Graph(id="wordcloud",figure={}),
                dbc.CardBody([
                    
                ])
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                    html.H3('Competitors performance'),
                    dcc.Graph(id="Scatter plot",figure={}),
                dbc.CardBody([
                   
                ])
            ]),
        ], width=7),

        
    
    ],className='mb-2'),
    
    dbc.Row([
        dbc.Col([
            
        ], width=1),
        dbc.Col([
            dbc.Card([
               
                dbc.CardBody([
                    html.H3('Age distribution'),
                    dcc.Graph(id="bar_knn",figure={})
                ])
            ]),
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H3('Score distribution'),
                    dcc.Dropdown(
                        id='sex',
                        options=[{'label': x, 'value': x}
                            for x in user['sex'].unique()],     
                        value='F',

                        style={'width' : '100%',
                            'text-align': 'center',
                            }
                        
                    ),
                    dcc.Graph(id="scatte",figure={})
                    
                ])
            ]),
        ], width=3),

        dbc.Col([
            dbc.Card([
               
                dbc.CardBody([
                    html.H3('Gender'),
                    dcc.Graph(id="sexpie",figure={})
                ])
            ]),
        ], width=5),
        
    ],className='mb-2'),

     dbc.Row([
        
        

        dbc.Col([
            dbc.Card([
               
                dbc.CardBody([
                    html.H3('Localisation'),
                    dcc.Graph(id="map",figure={})
                ])
            ]),
        ], width=6),
        dbc.Col([
            dbc.Card([
               
                dbc.CardBody([
                    html.H3('Rating trend'),
                    dcc.Graph(id="time_seri",figure={})
                ])
            ]),
        ], width=6),
        
    ],className='mb-2'),

], fluid=True)

#-------------------
@app.callback(
    
    Output('content-connections1','children'),
    [Input(component_id='brand_id', component_property='value')]
)

def leng(tweet):
    return(len(products.loc[products.brand==tweet]))

@app.callback(
    
    Output('content-connections2','children'),
    [Input(component_id='brand_id', component_property='value')]
)

def rat(tweet):
    ratings=ratingss.copy()
    ratings['user_avg_rating'] = ratings.groupby('product_id')['rating'].transform('mean').round(2)
    df=pd.merge(ratings,products,left_on='product_id',right_on='ID_product')
    df1=df[df['brand']=='Adidas']
    return(len(df1.rating))


#Worldcloud-----------------------------------------

@app.callback(
    
    Output('wordcloud','figure'),
    [Input(component_id='brand_id', component_property='value')]
)

def update_worldcloud( tweet):
     
    ratings=ratingss.copy()
    ratings['user_avg_rating'] = ratings.groupby('product_id')['rating'].transform('mean').round(2)
    df=pd.merge(ratings,products,left_on='product_id',right_on='ID_product')
    df1=df[df['brand']==tweet]
    df2=df1[['user_avg_rating','name']]
    df3=df2.drop_duplicates()
    df4=df3.sort_values(by=['user_avg_rating'],ascending=False)

    my_wordcloud = WordCloud(
        background_color='white',
        height=500
    ).generate(''.join( df4.name))

    fig_wordcloud = px.imshow(my_wordcloud, template='ggplot2')
    fig_wordcloud.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    fig_wordcloud.update_xaxes(visible=False)
    fig_wordcloud.update_yaxes(visible=False)

    return fig_wordcloud


@app.callback(
    
    Output('Scatter plot','figure'),
    [Input(component_id='name_id', component_property='value')], 
    [Input(component_id='brand_id', component_property='value')]
    
)

def branding(name,brand):
    ratings=ratingss.copy()
    ratings['user_avg_rating'] = ratings.groupby('product_id')['rating'].transform('mean').round(2)
    df=pd.merge(ratings,products,left_on='product_id',right_on='ID_product')
    df1=df[df['name']==name]

    df2=df1[['user_avg_rating','brand']]
    df2['Score']=df2['user_avg_rating']*20
    df3=df2.drop_duplicates()
    df4=df3.sort_values(by=['Score'],ascending=False)
    df4['rank']=[1, 2, 3, 4,5,6,7,8]
    rank=df4[df4.brand == brand]['rank'].values
    df5=df4[df4.brand==brand]
    df4['Brand']=df4['brand']
    
    if brand in df4[:5].Brand.values:
        
        fig = px.bar(df4[:5],
                     x="Score",
                     y='Brand',
                     #range_x=[0,100],
                     range_x=[0,100],
                     color='Brand'
                     )
    else:
        fig = px.bar(df4[:5],
                     x="Score",
                     y='Brand',
                     #range_x=[0,100],
                     range_x=[0,100],
                     color='Brand'
                     )
        fig.update_layout(
            height=500,
            title_text='The brand is ranked '+str(rank)[1:2]+', with a average rating of '+ str(df5.Score.values)[1:3]+'%'
        )
    return(fig)


@app.callback(
    
    Output('bar_knn','figure'),
    [Input(component_id='brand_id', component_property='value')],
    [Input('sex','value')]
)


def update_pie(tweet1,tweet2):
    ratings=ratingss.copy()
    ratings['user_avg_rating'] = ratings.groupby('product_id')['rating'].transform('mean').round(2)
    df=pd.merge(ratings,products,left_on='product_id',right_on='ID_product')
    df1=df[df['brand']==tweet1]
    df2=pd.merge(df1,user,on='user_id')
    dff=df2.loc[df2.sex==tweet2]
    fig_pie =px.pie(
        dff,
        values='age', 
        
        names=dff.age,
        hole=.5,
        color_discrete_sequence=['maroon','red','coral']
        )
    return(fig_pie)

@app.callback(
    
    Output('scatte','figure'),
    [Input(component_id='brand_id', component_property='value')],
    [Input('user_id','value')]
)

def update_pie(name,tweet):
    ratings=ratingss.copy()
    ratings['user_avg_rating'] = ratings.groupby('product_id')['rating'].transform('mean').round(2)
    df=pd.merge(ratings,products,left_on='product_id',right_on='ID_product')
    df1=df[df['brand']==name]
    df1['pourcen']=df1['rating']*20
    df2=pd.merge(df1,user,on='user_id')
    
    dff=df2.loc[df2['user_id']==tweet]
    fig_pie =px.pie(
            dff,
            values='pourcen', 
            names=dff.pourcen,
            hole=.5,
            color_discrete_sequence=['deeppink','darkviolet']
            )
    return(fig_pie)

@app.callback(
    
    Output('map','figure'),
    [Input(component_id='brand_id', component_property='value')],
    [Input('user_id','value')]
)

def map(name, tweet):
    ratings=ratingss.copy()
    ratings['user_avg_rating'] = ratings.groupby('product_id')['rating'].transform('mean').round(2)
    df=pd.merge(ratings,products,left_on='product_id',right_on='ID_product')
    df1=df[df['brand']==name]
    df2=pd.merge(df1,user,on='user_id')
    dff=df2.loc[df2.user_id==tweet]
    dff['count']=dff.groupby('iso_id')['iso_id'].transform(len)


    fig = px.scatter_geo(dff, locations="iso_id",hover_name="Residence",size='count',
                     projection="natural earth")
    return(fig)

@app.callback(
    
    Output('sexpie','figure'),
    [Input(component_id='brand_id', component_property='value')]
)

def update_pie(tweet1):
    ratings=ratingss.copy()
    ratings['user_avg_rating'] = ratings.groupby('product_id')['rating'].transform('mean').round(2)
    df=pd.merge(ratings,products,left_on='product_id',right_on='ID_product')
    df1=df[df['brand']==tweet1]
    dff=pd.merge(df1,user,on='user_id')
    dfff=dff['sex'].value_counts()
    fig_pie =px.pie(
            dfff,
            values='sex',
            names=dfff.index,
            hole=.5,
            color_discrete_sequence=['gold','lemonchiffon']
            )
    return(fig_pie)

@app.callback(
    
    Output('time_seri','figure'),
    [Input(component_id='brand_id', component_property='value')]
)

def time_serie(brand):
    times=time.copy()
    times=times[times['brand']==brand]
    times['pourcen']=times['rating']*20
    fig = go.Figure([go.Scatter(x=times['day'], y=times['pourcen'])])
    return(fig)

# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)