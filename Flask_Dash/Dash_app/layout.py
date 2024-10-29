from dash import html, dcc
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Container import Container

LOGO = "https://th.bing.com/th/id/OIP.GrU9y8BMAgsJzE1R8K020QHaE8?w=1380&h=920&rs=1&pid=ImgDetMain"
def create_layout(app):
    return dbc.Container([
        dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src=LOGO, height="30px")),
                                dbc.Col(dbc.NavbarBrand("Adey Innovations Inc.", className="ms-2")),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="#",
                        style={"textDecoration": "none"},
                    ),
                    dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                    dbc.Collapse(
                        [
                            dbc.Nav(
                                [
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Home", active=True, href="http://127.0.0.1:5000/home/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Model Performance", href="http://127.0.0.1:5000/model-performance/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Predict Transaction", href="http://127.0.0.1:5000/make-predictions/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Dashboard", href="http://127.0.0.1:5000/dashboard/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                ],
                                className="ms-auto",
                                navbar=True,
                            ),
                        ],
                        id="navbar-collapse",
                        is_open=False,
                        navbar=True,
                    ),
                ]
            ),
            color="dark",
            dark=True,
        ),    
        dbc.Row([
            html.H1("Adey Innovations Inc.", style={'textAlign': 'center', 'color': 'navy', 'font-size': '2.5em', 'margin-bottom': '10px'}),
            html.H2('Ecommerce Fraud Data Analysis', style={'textAlign': 'center', 'color': 'gray', 'font-size': '1.5em', 'font-style': 'italic', 'margin-bottom': '20px'}),
            html.Br()
        ], style={'marginBottom': '20px'}),
        dbc.Row([
            html.H4("Summary of Transactions and Fraud for Identified Countries", style={'color': 'darkblue', 'text-align': 'center', 'font-style': 'italic', 'margin-bottom': '20px'}),
            dcc.Interval(id='interval', interval=60*1000, n_intervals=0),
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        dbc.CardHeader("Total Transactions", className="text-center text-primary"),
                        html.H2(id="data-output", className="text-center text-info"),
                        # dbc.CardFooter('This calculation includes unknown countries', className="text-center text-muted")
                    ]),
                    className="p-3 shadow"
                )
            ], md=4),
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        dbc.CardHeader("Total Fraud Cases", className="text-center text-danger"),
                        html.H2(id="data-output2", className="text-center text-danger"),
                        # dbc.CardFooter('This calculation includes unknown countries', className="text-center text-muted")
                    ]),
                    className="p-3 shadow"
                )
            ], md=4),
            dbc.Col([
                dbc.Card(
                    dbc.CardBody([
                        dbc.CardHeader("Fraud Percentage", className="text-center text-warning"),
                        html.H2(id="data-output3", className="text-center text-warning"),
                        # dbc.CardFooter('This calculation includes unknown countries', className="text-center text-muted")
                    ]),
                    className="p-3 shadow"
                )
            ], md=4),
        ], style={'marginBottom': '20px'}),
        dbc.Row([
            html.H4("Summary of Transactions and Fraud Across Known and Unknown Countries", style={'color': 'darkblue', 'text-align': 'center', 'font-style': 'italic', 'margin-bottom': '20px'}),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.CardHeader("Total Trasactions", className="text-center text-primary"),
                        html.H2(id="data-output4", className="text-center text-info"),
                        # dbc.CardFooter('This calculation is only for trasactions with identified country')
                    ])
                ], className="p-3 shadow")
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.CardHeader("Total fraud cases", className="text-center text-danger"),
                        html.H2(id="data-output5", className="text-center text-danger"),
                        # dbc.CardFooter('This calculation is only for trasactions with identified country')
                    ])
                ], className="p-3 shadow")
            ], md=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.CardHeader("Fraud Percentage", className="text-center text-warning"),
                        html.H2(id="data-output6", className="text-center text-warning"),
                        # dbc.CardFooter('This calculation is only for trasactions with identified country')
                    ])
                ], className="p-3 shadow")
            ], md=4),
        ], style={'marginBottom': '20px'}),
        dbc.Row([
            dbc.Col([
                html.H4("Geographical Analysis", style={'color': 'darkblue', 'text-align': 'center', 'font-style': 'italic', 'margin-bottom': '20px'}),
                dcc.Graph(id='geographic-plots'),
            ], className="p-3 shadow")
        ]),
        dbc.Row([
            dbc.Col([
                html.H4("Browser Distribution of Fraud Cases", style={'color': 'darkblue', 'text-align': 'center', 'font-style': 'italic', 'margin-bottom': '20px'}),
                dcc.Graph(id='browser-fraud'),
            ],  className="p-3 shadow")
        ]),
        dbc.Row([
            dbc.Col([
                html.H4("Timeseries Analysis", style={'color': 'navy', 'text-align': 'center', 'margin-bottom': '20px'}),
                html.Div(
                    dcc.RadioItems(
                        id='time-value',
                        options=[
                            {'label': 'Date', 'value': 'Date'},
                            {'label': 'Month', 'value': 'Month'},
                            {'label': 'Day', 'value': 'Day'},
                            {'label': 'Hour', 'value': 'Hour'},
                            {'label': 'Minutes', 'value': 'Minutes'}
                        ],
                        value='Day',
                        inline=True,
                        style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '10px'},
                        labelStyle={'margin-right': '20px'}
                    ),
                ),
                dcc.Graph(id='timeseries-plots', style={'margin-top': '20px'})
            ], className="p-3 shadow", style={'background-color': 'white', 'border-radius': '10px', 'box-shadow': '0 0 10px rgba(0, 0, 0, 0.1)'})
        ]),
        dbc.Row([
            dbc.Col([
                html.H4("Device ID Distribution of Fraud Cases", style={'color': 'darkblue', 'text-align': 'center', 'font-style': 'italic', 'margin-bottom': '20px'}),
                dcc.Graph(id='device-distribution'),
                html.Div(dcc.Slider(
                    min=100,
                    max=3000,
                    step=500,
                    value=800,
                    id = 'row-slider',
                    
                ), style={'width': '49%', 'padding': '0px 20px 20px 20px'}),

            ], className="p-3 shadow")
        ]),
        dbc.Row([
            html.Br(),
            html.Br(),
            html.H4("Bivariant Analysis", style={'color': 'darkblue', 'text-align': 'center', 'font-style': 'italic', 'margin-bottom': '20px'}),
            dbc.Col([
                dcc.Graph(id='gender-distribution')
            ], md= 6, className="p-3 shadow"),
            dbc.Col([
                dcc.Graph(id='browser-distribution')
            ], md=6, className="p-3 shadow"),
        ]), 
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='source-distribution')
            ], md=6, className="p-3 shadow"),       
            dbc.Col([
                dcc.Graph(id='age-distribution')
            ], md=6, className="p-3 shadow")                      
        ]),
    ], fluid=True, style={'background-color': '#f7f7f7', 'padding': '20px'})


def create_home_layout(app):
    return dbc.Container([
        dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src=LOGO, height="30px")),
                                dbc.Col(dbc.NavbarBrand("Adey Innovations Inc.", className="ms-2")),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="#",
                        style={"textDecoration": "none"},
                    ),
                    dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                    dbc.Collapse(
                        [
                            dbc.Nav(
                                [
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Home", active=True, href="http://127.0.0.1:5000/home/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Model Performance", href="http://127.0.0.1:5000/model-performance/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Predict Transaction", href="http://127.0.0.1:5000/make-predictions/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Dashboard", href="http://127.0.0.1:5000/dashboard/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                ],
                                className="ms-auto",
                                navbar=True,
                            ),
                        ],
                        id="navbar-collapse",
                        is_open=False,
                        navbar=True,
                    ),
                ]
            ),
            color="dark",
            dark=True,
        ),    
        html.Div(
            [
                html.H1("Welcome to Adey Innovations Inc.", style={"text-align": "center", "color": "darkblue"}),
                html.Hr(),
                html.H2("Overview", style={"color": "darkblue"}),
                html.P(
                    "Adey Innovations Inc., a top company in the financial technology sector. "
                    "This project aims to create accurate and strong fraud detection models that handle the unique challenges of transaction data. "
                    "It also includes using geolocation analysis and transaction pattern recognition to improve detection."
                ),
                html.H2("Business Need", style={"color": "darkblue"}),
                html.P(
                    "Good fraud detection greatly improves transaction security. By using advanced machine learning models and detailed data analysis, Adey Innovations Inc. can spot fraudulent activities more accurately. "
                    "This helps prevent financial losses and builds trust with customers and financial institutions. A well-designed fraud detection system also makes real-time monitoring and reporting more efficient, allowing businesses to act quickly and reduce risks."
                ),
                html.P(
                    "This project will involve:",
                    style={"font-weight": "bold"}
                ),
                html.Ul([
                    html.Li("Analyzing and preprocessing transaction data."),
                    html.Li("Creating and engineering features that help identify fraud patterns."),
                    html.Li("Building and training machine learning models to detect fraud."),
                    html.Li("Evaluating model performance and making necessary improvements."),
                    html.Li("Deploying the models for real-time fraud detection and setting up monitoring for continuous improvement.")
                ]),
            ], style={'padding': '20px', 'color': 'black'}
        )    
    ], fluid=True, style={'padding': '20px'})



def create_predictions_layout(app):
    return dbc.Container([
        dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src=LOGO, height="30px")),
                                dbc.Col(dbc.NavbarBrand("Adey Innovations Inc.", className="ms-2")),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="#",
                        style={"textDecoration": "none"},
                    ),
                    dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                    dbc.Collapse(
                        [
                            dbc.Nav(
                                [
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Home", active=True, href="http://127.0.0.1:5000/home/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Model Performance", href="http://127.0.0.1:5000/model-performance/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Predict Transaction", href="http://127.0.0.1:5000/make-predictions/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Dashboard", href="http://127.0.0.1:5000/dashboard/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                ],
                                className="ms-auto",
                                navbar=True,
                            ),
                        ],
                        id="navbar-collapse",
                        is_open=False,
                        navbar=True,
                    ),
                ]
            ),
            color="dark",
            dark=True,
        ),        
        html.Div(
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.H3('Enter User Data:', style={'text-align': 'center', 'color': 'darkblue'}),
                            html.Div(
                                [
                                    html.Label('Purchase Value:', style={'margin-top': '10px'}),
                                    dcc.Input(id='purchase_value', type='number', value=46, style={'width': '100%', 'padding': '8px', 'margin-bottom': '10px'}),
                                    html.Label('Age:', style={'margin-top': '10px'}),
                                    dcc.Input(id='age', type='number', value=34, style={'width': '100%', 'padding': '8px', 'margin-bottom': '10px'}),
                                    html.Label('Purchase Time:', style={'margin-top': '10px'}),
                                    dcc.Input(id='purchase_time', type='datetime-local', value='2015-04-18T02:47', style={'width': '100%', 'padding': '8px', 'margin-bottom': '10px'}),
                                    html.Label('Signup Time:', style={'margin-top': '10px'}),
                                    dcc.Input(id='signup_time', type='datetime-local', value='2015-02-18T22:55', style={'width': '100%', 'padding': '8px', 'margin-bottom': '10px'}),
                                    html.Label('Source:', style={'margin-top': '10px'}),
                                    dcc.Input(id='source', type='text', placeholder='Enter the source', value='SEO', style={'width': '100%', 'padding': '8px', 'margin-bottom': '10px'}),
                                    html.Label('Browser:', style={'margin-top': '10px'}),
                                    dcc.Input(id='browser', type='text', placeholder='Enter the browser type', value='Safari', style={'width': '100%', 'padding': '8px', 'margin-bottom': '10px'}),
                                    html.Label('Sex:', style={'margin-top': '10px'}),
                                    dcc.Input(id='sex', type='text', placeholder='Enter (M) for male or (F) for female', value='M', style={'width': '100%', 'padding': '8px', 'margin-bottom': '10px'}),
                                    html.Div(className='input-line', children=[
                                        html.Label('Transaction Frequency:', style={'padding-top': '10px'}),
                                        dcc.Input(id='trasaction_frequency', type='number', placeholder='Enter the total transactions for a user', value=5, style={'flex': '1', 'border': 'none', 'border-bottom': '1px solid #888', 'padding': '8px'}),
                                    ]),
                                    html.Label('Country:', style={'margin-top': '10px'}),
                                    dcc.Input(id='country', type='text', placeholder='Enter the country', value='Angola', style={'width': '100%', 'padding': '8px', 'margin-bottom': '10px'}),
                                ],
                                style={'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px'}
                            ),
                            html.Button('Predict', id='predict-button', style={'margin-top': '20px', 'background-color': 'darkblue', 'color': 'white', 'border-radius': '5px'}),
                            html.Div(id='output', style={'margin-top': '20px', 'text-align': 'center', 'font-weight': 'bold'})
                        ],
                        style={'max-width': '600px', 'margin': 'auto'}
                    )
                )
            )
        )
    ], fluid=True, style={'padding': '20px'})     


def create_model_layout(app):
    return dbc.Container([
        dbc.Navbar(
            dbc.Container(
                [
                    html.A(
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src=LOGO, height="30px")),
                                dbc.Col(dbc.NavbarBrand("Adey Innovations Inc.", className="ms-2")),
                            ],
                            align="center",
                            className="g-0",
                        ),
                        href="#",
                        style={"textDecoration": "none"},
                    ),
                    dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                    dbc.Collapse(
                        [
                            dbc.Nav(
                                [
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Home", active=True, href="http://127.0.0.1:5000/home/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Model Performance", href="http://127.0.0.1:5000/model-performance/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Predict Transaction", href="http://127.0.0.1:5000/make-predictions/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                    html.Div(
                                        dbc.NavItem(dbc.NavLink("Dashboard", href="http://127.0.0.1:5000/dashboard/")),
                                        style={"background-color": "darkblue", "color": "white", "padding": "10px", "border-radius": "5px", "margin-right": "10px"}
                                    ),
                                ],
                                className="ms-auto",
                                navbar=True,
                            ),
                        ],
                        id="navbar-collapse",
                        is_open=False,
                        navbar=True,
                    ),
                ]
            ),
            color="dark",
            dark=True,
        ),    
        dbc.Row([
            dcc.Interval(id='interval', interval=10*1000, n_intervals=0),
            dbc.Col([
                dcc.Graph(id='epochs')
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.H4("Model metrics evaluation", style={'color': 'navy', 'text-align': 'center', 'margin-bottom': '20px'}),
                    html.Div(
                        dcc.RadioItems(
                            id='metrics-value',
                            options=[
                                {'label': 'AUC-ROC', 'value': 'auc_roc'},
                                {'label': 'Precision', 'value': 'precision'},
                                {'label': 'Recall', 'value': 'recall'},
                                {'label': 'f1', 'value': 'f1'},
                            ],
                            value='precision',
                            inline=True,
                            style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '10px'},
                            labelStyle={'margin-right': '20px'}
                    ),
                ),
                dcc.Graph(id='metrics-plots', style={'margin-top': '20px'})
            ], className="p-3 shadow", style={'background-color': 'white', 'border-radius': '10px', 'box-shadow': '0 0 10px rgba(0, 0, 0, 0.1)'})
        ]),      
        dbc.Row([
            dbc.Col([
                html.H4("Neural network Model metrics evaluation", style={'color': 'navy', 'text-align': 'center', 'margin-bottom': '20px'}),
                    html.Div(
                        dcc.RadioItems(
                            id='neruron-metrics-value',
                            options=[
                                {'label': 'Validation Accuracy', 'value': 'val_accuracy'},
                                {'label': 'Validation Precision', 'value': 'val_precision'},
                                {'label': 'Validation Recall', 'value': 'val_recall'},
                                {'label': 'Validation f1', 'value': 'val_f1_score'},
                                {'label': 'Validation Loss', 'value': 'val_loss'},
                                {'label': 'Accuracy', 'value': 'accuracy'},
                                {'label': 'Precision', 'value': 'precision'},
                                {'label': 'Recall', 'value': 'recall'},
                                {'label': 'f1', 'value': 'f1_score'},
                                {'label': 'Loss', 'value': 'loss'},                                
                            ],
                            value='precision',
                            inline=True,
                            style={'display': 'flex', 'justify-content': 'space-between', 'margin-top': '10px'},
                            labelStyle={'margin-right': '20px'}
                    ),
                ),                           
                dcc.Graph(id='metrics-neuron', style={'margin-top': '20px'})
            ], className="p-3 shadow", style={'background-color': 'white', 'border-radius': '10px', 'box-shadow': '0 0 10px rgba(0, 0, 0, 0.1)'})
        ]),              
    ], fluid=True, style={'padding': '20px'})          