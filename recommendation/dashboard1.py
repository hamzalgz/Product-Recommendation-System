#library
import os
import pathlib
import numpy as np

from dash.dependencies import Input, Output
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
url_11='https://assets6.lottiefiles.com/private_files/lf30_2c7wnifx.json'
url_22='https://assets5.lottiefiles.com/packages/lf20_ur7sluxh.json'
url_33='https://assets2.lottiefiles.com/packages/lf20_Y9BNoF.json'


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SKETCHY],
)


def build_tabs():
    return html.Div(
        id="tabs",
        className="tabs",
        children=[
            dcc.Tabs(
                id="app-tabs",
                value="tab2",
                className="custom-tabs",
                children=[
                    dcc.Tab(
                        id="user-tab",
                        label="the user",
                        value="tab1",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=[
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
                                            html.H6('Performance'),
                                            html.H2(id='help')
                                        ], style={'textAlign':'center'})
                                    ]),
                                ], width=2),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_11)),
                                        dbc.CardBody([
                                            html.H6('Picked'),
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
                                            html.H3("Avg score"),
                                            
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
                                            html.H3("Rated products"),
                                            html.Div(id='none2',children=[]),
                                            dcc.Graph(id="histograme"  ,figure={}),
                                        ])
                                    ]),
                                ], width=5),
                                dbc.Col([
                                    
                                ], width=1),
                                
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
                                    
                                ], width=2),
                                
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H3("Localisation"),
                                            
                                            dcc.Graph(id="graph")
                                            
                                        ])
                                    ]),
                                ], width=7),
                                
                            ],className='mb-2'),

                        ]
                        
                    ),
                    dcc.Tab(
                        id="commercial-tab",
                        label="commercial",
                        value="tab2",
                        className="custom-tab",
                        selected_className="custom-tab--selected",
                        children=[
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
                                            html.H6('Performance'),
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
                                            html.H3('Most popular products'),
                                            dcc.Graph(id="wordcloud_bar",figure={}),
                                        dbc.CardBody([
                                            
                                        ])
                                    ]),
                                ], width=8),
                                

                                
                            
                            ],className='mb-2'),
                            dbc.Row([

                                dbc.Col([
                                    
                                ], width=2),
                                dbc.Col([
                                    dbc.Card([
                                            html.H3('Competitors performance'),
                                            dcc.Graph(id="Scatter plot",figure={}),
                                        dbc.CardBody([
                                        
                                        ])
                                    ]),
                                ], width=9),
                                
                                
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
                            
                        ]
                        
                    ),


                   
                ],
            )
        ],
    )





app.layout =dbc.Container([
    dcc.Markdown(
            '## Recommendation system'
            ),
    build_tabs(),
    
], fluid=True)


 #Histogramm1-----------------------------------------------   

#---------------

#-------------------

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
        a=df4[df4.Brand=='Adidas']['rank'].values[0]
        dfff={}
        dfff['Brand']=['Competitor 1','Competitor 2','Competitor 3','Competitor 4','Competitor 5']

        dfff['Score']=df4['Score'].tolist()[:5]
        dfff['Brand'][a-1]=brand
        
        dfff=pd.DataFrame(dfff)
        fig = px.bar(dfff,
                     x="Score",
                     y='Brand',
                     #range_x=[0,100],
                     range_x=[0,100],
                     color='Brand'
                     )
    else:
        dfff={}
        dfff['Brand']=['Competitor 1','Competitor 2','Competitor 3','Competitor 4','Competitor 5']

        dfff['Score']=df4['Score'].tolist()[:5]

        dfff=pd.DataFrame(dfff)
        fig = px.bar(dfff,
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
    
    Output('wordcloud_bar','figure'),
    [Input(component_id='brand_id', component_property='value')]
)



def barrr( tweet):
     
    ratings=ratingss.copy()
    ratings['user_avg_rating'] = ratings.groupby('product_id')['rating'].transform('mean').round(2)
    df=pd.merge(ratings,products,left_on='product_id',right_on='ID_product')
    df1=df[df['brand']==tweet]
    df1['Score']=df1['user_avg_rating']*20
    df2=df1[['Score','name']]
    df3=df2.drop_duplicates()
    df3['Name']=df['name']
    

    fig = px.bar(df3,
                x='Score',
                 y="Name",
                 
                 #range_x=[0,100],
                 range_x=[0,100],
                 color='Name'
         )

    return fig

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
    ratings=ratings[ratings['user_id']==5]
    new=pd.merge(ratings,products,left_on='product_id',right_on='ID_product')
    di={}
    di={'Type':['Products', 'Service', 'Travel', 'Random'],'Count':[0,0,0,0]}

    df=pd.DataFrame(di)
    df['Count'][0]=len(new[new['type']=='Products'])
    df['Count'][1]=len(new[new['type']=='Service'])
    df['Count'][2]=len(new[new['type']=='Travel'])
    df['Count'][3]=len(new[new['type']=='Random'])
    fig=px.bar(df, x="Type",y='Count',color_discrete_sequence=['darkviolet'])

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