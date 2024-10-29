from dash import html, dcc
import dash_bootstrap_components as dbc

def create_layout(app):
    return dbc.Container([
        dbc.Row([
            html.H1("Adey Innovations Inc.", style={'textAlign': 'center'}),
            html.H2('Ecommerce Fraud data analysis', style={'textAlign': 'center'}),
            html.Br()
        ], style={'marginBottom': '20px'}),
        dbc.Row([
            html.H3("Transaction and fraud summary for all known and unknown countries"),
            dcc.Interval(id='interval', interval=60*1000, n_intervals=0),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.CardHeader("Total Trasactions"),
                        dbc.CardBody(id="data-output"),
                        dbc.CardFooter('This calculation includes unknown countries')
                    ])
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.CardHeader("Total fraud cases"),
                        dbc.CardBody(id="data-output2"),
                        dbc.CardFooter('This calculation includes unknown countries')
                    ])
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.CardHeader("Fraud Percentage"),
                        dbc.CardBody(id="data-output3"),
                        dbc.CardFooter('This calculation includes unknown countries')
                    ])
                ])
            ], md=4),
        ], style={'marginBottom': '20px'}),
        dbc.Row([
            html.H3("Transaction and fraud summary for known countires countries"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.CardHeader("Total Trasactions"),
                        dbc.CardBody(id="data-output4"),
                        dbc.CardFooter('This calculation is only for trasactions with identified country')
                    ])
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.CardHeader("Total fraud cases"),
                        dbc.CardBody(id="data-output5"),
                        dbc.CardFooter('This calculation is only for trasactions with identified country')
                    ])
                ])
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.CardHeader("Fraud Percentage"),
                        dbc.CardBody(id="data-output6"),
                        dbc.CardFooter('This calculation is only for trasactions with identified country')
                    ])
                ])
            ], md=4),
        ], style={'marginBottom': '20px'}),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='geographic-plots'),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='browser-fraud'),
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.H3("Timeseries analysis"),
                html.Div(dcc.RadioItems(
                    ['Date', 'Month', 'Day', 'Hour', 'Minutes'],
                    'Day',
                    id='time-value', 
                    inline=True,
                ), style={"margin-right": "30px"}),
                dcc.Graph(id='timeseries-plots')
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='device-distribution'),
                html.Div(dcc.Slider(
                    min=100,
                    max=3000,
                    step=500,
                    value=800,
                    id = 'row-slider',
                    
                ), style={'width': '49%', 'padding': '0px 20px 20px 20px'}),

            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='gender-distribution')
            ], md= 6),
            dbc.Col([
                dcc.Graph(id='browser-distribution')
            ], md=6),
        ]), 
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='source-distribution')
            ], md=6),       
            dbc.Col([
                dcc.Graph(id='age-distribution')
            ], md=6)                      
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H3('Enter User Data:'),
                    html.Label('Purchase Value:'),
                    dcc.Input(id='purchase_value', type='number', value=46),
                    html.Label('Age:'),
                    dcc.Input(id='age', type='number', value=34),
                    html.Label('Purchase time:'),
                    dcc.Input(id='purchase_time', type='datetime-local', value='2015-04-18 02:47:11'),      
                    html.Label('Signup time:'),
                    dcc.Input(id='signup_time', type='datetime-local', value='2015-02-18 22:55:49'),      
                    html.Label('Source:'),
                    dcc.Input(id='source', type='text', placeholder='Enter the source', value='SEO'),    
                    html.Label('Browser:'),
                    dcc.Input(id='browser', type='text', placeholder='Enter the browser type', value='Safari'),    
                    html.Label('Sex:'),
                    dcc.Input(id='sex', type='text', placeholder='Enter the (M) for male and (F) for female', value='M'),           
                    html.Label('Trasaction Frequency:'),
                    dcc.Input(id='trasaction_frequency', type='number', placeholder='Enter the total trasaction for a user', value=5),         
                    html.Label('Country:'),
                    dcc.Input(id='country', type='text', placeholder='Enter the country', value='Angola'),                                                                                           

                    html.Button('Predict', id='predict-button'),
                    html.Div(id='output')
                ])
            ])
        ])
    ], fluid=True)