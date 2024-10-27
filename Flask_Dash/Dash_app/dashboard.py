# dashboard.py
import dash
from dash import html

def create_dash_app(flask_app):
    app = dash.Dash(
        __name__,
        server=flask_app,  
        routes_pathname_prefix='/app1/'  
    )
    app.layout = html.Div("Dash app 1")
    return app
