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
                dcc.RadioItems(
                    ['Date', 'Month', 'Day', 'Hour', 'Minutes'],
                    'Day',
                    id='time-value', 
                    inline=True,
                ),
                dcc.Graph(id='timeseries-plots')
            ])
        ]),

    ], fluid=True)