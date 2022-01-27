#library
import os
import pathlib
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

customer = pd.read_csv('D:/retour/cluster code/customer.csv')

users = pd.read_csv('D:/retour/recommendation/user1.csv', encoding='latin-1')

ratings = pd.read_csv('D:/retour/recommendation/ratingg.csv', encoding='latin-1')

products = pd.read_csv('D:/retour/recommendation/products1.csv', encoding='latin-1')

ratings['user_avg_rating'] = ratings.groupby('user_id')['rating'].transform('mean').round(2)
rate = pd.merge(ratings, users, on='user_id')

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

def get_rating_(userid, productid):
    return ratings.loc[(ratings.user_id == userid) & (ratings.product_id == productid), 'rating'].iloc[0]

def get_productids_(userid):
    return ratings.loc[(ratings.user_id == userid), 'product_id'].tolist()

def get_product_name_(productid):
    return products.loc[(products.product_id == productid), 'product_name'].iloc[0]

def distance_similarity_score(user1, user2):
    """
    distance between the two users (i.e. Euclidean distances)
    """
    both_buy_count = 0
    for element in ratings.loc[ratings.user_id == user1, 'product_id'].tolist():
        if element in ratings.loc[ratings.user_id == user2, 'product_id'].tolist():
            both_buy_count += 1
    if both_buy_count == 0:
        return 0
    distance = []
    for element in ratings.loc[ratings.user_id == user1, 'product_id'].tolist():
        if element in ratings.loc[ratings.user_id == user2, 'product_id'].tolist():
            rating1 = get_rating_(user1, element)
            rating2 = get_rating_(user2, element)
            distance.append(pow(rating1 - rating2, 2))
    total_distance = np.sum(distance)
    return 1 / (1 + sqrt(total_distance))

def pearson_correlation_score(user1, user2):
    """
    user1 & user2 : user ids of two users between which similarity score is to be calculated.
    """
    both_buy_count = []
    for element in ratings.loc[ratings.user_id == user1, 'product_id'].tolist():
        if element in ratings.loc[ratings.user_id == user2, 'product_id'].tolist():
            both_buy_count.append(element)
    if len(both_buy_count) == 0:
        return 0
    rating_sum_1 = np.sum([get_rating_(user1, element) for element in both_buy_count])
    rating_sum_2 = np.sum([get_rating_(user2, element) for element in both_buy_count])
    rating_squared_sum_1 = np.sum([pow(get_rating_(user1, element), 2) for element in both_buy_count])
    rating_squared_sum_2 = np.sum([pow(get_rating_(user2, element), 2) for element in both_buy_count])
    product_sum_rating = np.sum([get_rating_(user1, element) * get_rating_(user2, element) for element in both_buy_count])

    numerator = product_sum_rating - ((rating_sum_1 * rating_sum_2) / len(both_buy_count))
    denominator = sqrt((rating_squared_sum_1 - pow(rating_sum_1, 2) / len(both_buy_count)) * (
            rating_squared_sum_2 - pow(rating_sum_2, 2) / len(both_buy_count)))
    if denominator == 0:
        return 0
    return numerator / denominator

# Most similar users using two scores i.e., Euclidean distance and pearson correlation.
def most_similar_users_(user1, number_of_users, metric='pearson'):
    # Getting distinct user ids.
    user_ids = ratings.user_id.unique().tolist()

    # Getting similarity score between targeted and every other user in the list(or subset of the list).
    if metric == 'pearson':
        similarity_score = [(pearson_correlation_score(user1, nth_user), nth_user) for nth_user in user_ids[:30] if
                            nth_user != user1]
    else:
        similarity_score = [(distance_similarity_score(user1, nth_user), nth_user) for nth_user in user_ids[:30] if
                            nth_user != user1]

    # Sorting in descending order.
    similarity_score.sort()
    similarity_score.reverse()

    # Returning the top most 'number_of_users' similar users.
    return similarity_score[:number_of_users]


def get_recommendation_(userid):
    user_ids = ratings.user_id.unique().tolist()
    total = {}
    similariy_sum = {}

    # Iterating over subset of user ids.
    for user in user_ids[:30]:

        # not comparing the user to itself
        if user == userid:
            continue

        # Getting similarity score between the users.
        score = pearson_correlation_score(userid, user)

        # not considering users having zero or less similarity score.
        if score <= 0:
            continue

        # Getting weighted similarity score and sum of similarities between both the users.
        for productid in get_productids_(user):
            # Only considering not bought/not rated products
            if productid not in get_productids_(userid) or get_rating_(userid, productid) == 0:
                total[productid] = 0
                total[productid] += get_rating_(user, productid) * score
                similariy_sum[productid] = 0
                similariy_sum[productid] += score

    # Normalizing ratings
    ranking = [(tot / similariy_sum[productid], productid) for productid, tot in total.items()]
    ranking.sort()
    ranking.reverse()

    # Getting product names against the product ids.
    recommendations = [get_product_name_(productid) for score, productid in ranking]
    return recommendations[:10]

ratedproducts = pd.merge(ratings, products, on='product_id')

if not ratedproducts[ratedproducts.duplicated(['user_id', 'product_name'])].empty:
    new_ratings = ratedproducts.drop_duplicates(['user_id', 'product_name'])

df_for_knn = new_ratings.pivot(index='user_id', columns='product_name', values='rating').fillna(0)
print(df_for_knn.head())
df_for_knn_sparse = csr_matrix(df_for_knn.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(df_for_knn_sparse)

userid = 5
n_neighbors = 6  # for top 5 (the first is the user itself)
distances, indices = model_knn.kneighbors(df_for_knn.loc[userid].values.reshape(1, -1),
                                          n_neighbors=6)  # get the top k similar users to this particular user
print("Nearest Neighbors of User with ID = {0}:".format(userid))
for i in range(0, len(distances.flatten())):
    if i != 0:  # avoid the first user that is the user itself
        print("{0}: UserId: {1} -- Distance: {2}".format(i, df_for_knn.index[indices.flatten()[i]],
                                                         distances.flatten()[i]))

sum = 0
weightsum = 0
product_name = 'FruitCake'  # the product we'll try to predict the rating
for i in range(0, len(distances.flatten())):
    if i != 0:
        indice = df_for_knn.index[indices.flatten()[i]]
        weight = 1 - distances.flatten()[i]  # computes the weights according to the similarity
        sum = sum + df_for_knn.loc[indice, product_name] * weight  # computes the weighted sum
        weightsum = weightsum + weight
print("The rating that the user with ID = {0} will give in the product {1} is: {2}".format(userid, product_name,
                                                                                           round(sum / weightsum, 1)))

for j in range(0, len(distances.flatten())):
    if j != 0:
        indice = df_for_knn.index[indices.flatten()[j]]


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
                                            html.H6('users'),
                                            html.H2(id='content-connections33', children=len(users.user_id))
                                        ], style={'textAlign':'center'})
                                    ]),
                                ], width=2),
                                
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_22)),
                                        dbc.CardBody([
                                            html.H6('ratings'),
                                            html.H2(id='content-connections22', children=len(ratings.rating))
                                        ], style={'textAlign':'center'})
                                    ]),
                                ], width=2),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_11)),
                                        dbc.CardBody([
                                            html.H6('products'),
                                            html.H2(id='content-connections11', children=len(products.product_id))
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
                                            html.H3("histogram of user_id by the avrage of rating"),
                                            
                                            html.Div(id='none1',children=[]),
                                            dcc.Graph(id="histog"  ,figure={}),
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
                                            html.H3("pie-chart of age by sex"),
                                            dcc.Graph(id="pie-chart"  ,figure={}),
                                        ])
                                    ]),
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        
                                        dbc.CardBody([
                                            html.H3("pie-chart of rating"),
                                            html.Div(id='none',children=[]),
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
                                            html.H3("bar of cartegory "),
                                            html.Div(id='none2',children=[]),
                                            dcc.Graph(id="histograme"  ,figure={}),
                                        ])
                                    ]),
                                ], width=4),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H3("distribution of user_id by sex"),
                                            html.P("Select Distribution:"),
                                            dcc.RadioItems(
                                                id='dist-marginal',
                                                options=[
                                                    {'label': ' box       __', 'value': 'box'},
                                                    {'label': '  violin      __', 'value': 'violin'},
                                                    {'label': '    rug', 'value': 'rug'}

                                                ],
                                                value='box',
                                                style={"padding": "10px", "max-width": "800px", "margin": "auto"},
                                            ),
                                            dcc.Dropdown(
                                                id='gender',
                                                options=[{'label': 'Female', 'value': 'F'},
                                                        {'label': 'Male', 'value': 'M'},
                                                    ],
                                                value='F',

                                                style={'width' : '90%',
                                                    'text-align': 'center',
                                                    }
                                                
                                            ),
                                            dcc.Graph(id="graph")
                                            
                                        ])
                                    ]),
                                ], width=6),
                                
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
                                    
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_3)),
                                        dbc.CardBody([
                                            html.H6('customer'),
                                            html.H2(id='content-connections3', children=len(customer['CustomerID'].unique()))
                                        ], style={'textAlign':'center'})
                                    ]),
                                ], width=2),
                                
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_1)),
                                        dbc.CardBody([
                                            html.H6('products'),
                                            html.H2(id='content-connections1', children=len(customer['ProductID'].unique()))
                                        ], style={'textAlign': 'center'})
                                    ]),
                                ], width=2),

                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardHeader(Lottie(options=options, width="100%", height="100%", url=url_2)),
                                        dbc.CardBody([
                                            html.H6('ratings'),
                                            html.H2(id='content-connections2', children=len(customer[' rating']))
                                        ], style={'textAlign':'center'})
                                    ]),
                                ], width=2),
                                
                                
                            ],className='mb-2'),
                            dbc.Row([
                                dbc.Col([
                                    
                                ], width=1),
                                
                                dbc.Col([
                                    dbc.Card([
                                            html.H3('wordcloud of recommendation products for each user_id'),
                                            dcc.Graph(id="wordcloud",figure={}),
                                        dbc.CardBody([
                                            
                                        ])
                                    ]),
                                ], width=3),
                                dbc.Col([
                                    dbc.Card([
                                            html.H3('Scatter plot the top most 10 similar users'),
                                            dcc.Graph(id="Scatter plot",figure={}),
                                        dbc.CardBody([
                                        
                                        ])
                                    ]),
                                ], width=4),

                                dbc.Col([
                                    dbc.Card([
                                    html.H3('choose an user_id'),
                                        dcc.Dropdown(
                                            id='id',
                                            options=[{'label': x, 'value': x}
                                                for x in customer['CustomerID'].unique()],     
                                            value=5,

                                            style={'width' : '100%',
                                                'text-align': 'center',
                                                }
                                            
                                        ),
                                    ]),
                                ], width=3),
                            
                            ],className='mb-2'),
                            
                            dbc.Row([
                                dbc.Col([
                                    
                                ], width=1),
                                dbc.Col([
                                    dbc.Card([
                                    
                                        dbc.CardBody([
                                            html.H3('predicting the rate of each category based on user_id'),
                                            dcc.Graph(id="bar_knn",figure={})
                                        ])
                                    ]),
                                ], width=5),
                                dbc.Col([
                                    dbc.Card([
                                        dbc.CardBody([
                                            html.H3('distance between our user and the nears user using KNN'),
                                            dcc.Graph(id="scatte",figure={})
                                            
                                        ])
                                    ]),
                                ], width=5),
                                
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


 #Histogramm1-----------------------------------------------   
figure={}
@app.callback(
    Output('histog','figure'),
    [Input('none1','children')]
)
def update_histogramm(input):
    ratings['user_avg_rating'] = ratings.groupby('user_id')['rating'].transform('mean').round(2)
    fig = px.bar(ratings,
                 x="user_id",
                 y="user_avg_rating"
                 )
    return(fig)

#pie-------------------------
@app.callback(
    Output(component_id='pie-chart',component_property='figure'),
    [Input('len','value')])

def update_pie(tweet):
      
    dff=users.loc[users.sex==tweet]
    fig_pie =px.pie(
        dff,
        values='age', 
        
        names=dff.age,
        hole=.5,
        color_discrete_sequence=['maroon','red','coral']
        )
    return(fig_pie)

#---------------------------------
@app.callback(
    Output(component_id='pie-chart1',component_property='figure'),
    [Input('none','children')])

def update_pie(tweet):

    fig_pie =px.pie(
        ratings,
        values='rating', 
        
        names=ratings.rating,
        hole=.5,
        color_discrete_sequence=['deeppink','darkviolet']
        )
    return(fig_pie)

@app.callback(
    Output(component_id='histograme',component_property='figure'),
    [Input('none2','children')])
def update_pie(tweet):

    fig=px.histogram(products, x="category",color_discrete_sequence=['darkviolet'])
    return(fig)

@app.callback(
    Output("graph", "figure"), 
    [Input("dist-marginal", "value")],[Input("gender", "value")])

def display_graph(marginal,gender):
    dff=rate.loc[rate.sex==gender]
    fig = px.histogram(
        dff, x="user_id",color='sex',
        marginal=marginal, range_x=[0,30],
        hover_data=ratings.columns,
         color_discrete_sequence=['deeppink','darkviolet'])

    return fig


@app.callback(
    
    Output('wordcloud','figure'),
    [Input(component_id='id', component_property='value')]
)

def update_worldcloud( tweet):
     
    user_ids = ratings.user_id.unique().tolist()
    total = {}
    similariy_sum = {}

    # Iterating over subset of user ids.
    for user in user_ids[:30]:

        # not comparing the user to itself
        if user == userid:
            continue

        # Getting similarity score between the users.
        score = pearson_correlation_score(userid, user)

        # not considering users having zero or less similarity score.
        if score <= 0:
            continue

        # Getting weighted similarity score and sum of similarities between both the users.
        for productid in get_productids_(user):
            # Only considering not bought/not rated products
            if productid not in get_productids_(userid) or get_rating_(userid, productid) == 0:
                total[productid] = 0
                total[productid] += get_rating_(user, productid) * score
                similariy_sum[productid] = 0
                similariy_sum[productid] += score

    # Normalizing ratings
    ranking = [(tot / similariy_sum[productid], productid) for productid, tot in total.items()]
    ranking.sort()
    ranking.reverse()

    # Getting product names against the product ids.
    recommendations = [get_product_name_(productid) for score, productid in ranking]
    recommendations[:10]

    my_wordcloud = WordCloud(
        background_color='white',
        height=500
    ).generate(" ".join( recommendations[:20]))

    fig_wordcloud = px.imshow(my_wordcloud, template='ggplot2')
    fig_wordcloud.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    fig_wordcloud.update_xaxes(visible=False)
    fig_wordcloud.update_yaxes(visible=False)

    return fig_wordcloud


@app.callback(
    
    Output('Scatter plot','figure'),
    [Input(component_id='id', component_property='value')]
)

def similar_user(tweet):
    user_ids = ratings.user_id.unique().tolist()
    similarity_score = [(distance_similarity_score(tweet, nth_user), nth_user) for nth_user in user_ids[:30] if
                            nth_user != tweet]

    # Sorting in descending order.
    similarity_score.sort()
    similarity_score.reverse()
    l=similarity_score[:10]
    distance=[]
    user_id=[]
    for i in range(10):
        distance.append(l[i][0])
        user_id.append(l[i][1])

    fig = px.scatter(x=distance,y=user_id,size=distance,color=distance)


    return(fig)


@app.callback(
    
    Output('bar_knn','figure'),
    [Input(component_id='id', component_property='value')]
)

def predict_rating(tweet):
    sum = 0
    weightsum = 0
    l=[]
    category=['Alcohol', 'Vegetable', 'Cake', 'Milk', 'Sauce', 'Fruit', 'Drink',
       'Grain', 'Fish', 'Chocolate', 'Berries']
    ratedproducts = pd.merge(ratings, products, on='product_id')
    if not ratedproducts[ratedproducts.duplicated(['user_id', 'category'])].empty:
        new_ratings = ratedproducts.drop_duplicates(['user_id', 'category'])
    df_for_knn = new_ratings.pivot(index='user_id', columns='category', values='rating').fillna(0)
    
    df_for_knn_sparse = csr_matrix(df_for_knn.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(df_for_knn_sparse)

    userid = tweet

    distances, indices = model_knn.kneighbors(df_for_knn.loc[userid].values.reshape(1, -1),
                                              n_neighbors=6) 
    
    for ca in category:
        product_name = ca # the product we'll try to predict the rating
        for i in range(0, len(distances.flatten())):
            if i != 0:
                indice = df_for_knn.index[indices.flatten()[i]]
                weight = 1 - distances.flatten()[i]  # computes the weights according to the similarity
                sum = sum + df_for_knn.loc[indice, product_name] * weight  # computes the weighted sum
                weightsum = weightsum + weight


        l.append(round(sum / weightsum, 1))
        

    fig = px.bar( y=category, x=l,range_x=[0,5],color=category)
    return(fig)



@app.callback(
    
    Output('scatte','figure'),
    [Input(component_id='id', component_property='value')]
)

def distancee(tweet):

    if not ratedproducts[ratedproducts.duplicated(['user_id', 'category'])].empty:
        new_ratings = ratedproducts.drop_duplicates(['user_id', 'category'])
    df_for_knn = new_ratings.pivot(index='user_id', columns='category', values='rating').fillna(0)
    
    df_for_knn_sparse = csr_matrix(df_for_knn.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(df_for_knn_sparse)

    userid = tweet

    distances, indices = model_knn.kneighbors(df_for_knn.loc[userid].values.reshape(1, -1),
                                              n_neighbors=7)
    user_id=[]
    distance=[]
    for i in range(0, len(distances.flatten())):
        user_id.append(df_for_knn.index[indices.flatten()[i]])
        distance.append(distances.flatten()[i])
    fig = px.scatter(x=distance,y=user_id,size=distance,color=distance)
    return(fig)




# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)