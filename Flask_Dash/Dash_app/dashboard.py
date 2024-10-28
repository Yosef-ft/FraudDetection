import dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_bootstrap_components as dbc

import requests
import pandas as pd
import geopandas as gpd
import plotly.express as px


from .layout import create_layout
from Flask_app.utils import Utils

def create_dash_app(flask_app):

    utils = Utils()

    app = dash.Dash(
        __name__,
        server=flask_app,
        routes_pathname_prefix='/app1/',
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )
    
    app.layout = create_layout(app)


    @app.callback(Output("data-output", "children"), 
                  Output("data-output2", "children"),
                  Output("data-output3", "children"),
                  [Input("interval", "n_intervals")])
    def fetch_data(n):
        # Getting data from flask
        response = requests.get("http://127.0.0.1:5000/data") 
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            total_transaction, fraud_cases, fraud_percentage = utils.trasaction_summary(data, include_other=False)
            return html.Div(f"{total_transaction}"), html.Div(f" {fraud_cases}"), html.Div(f"{fraud_percentage} %")
        else:
            return html.Div("Failed to load data")         



    @app.callback(Output("data-output4", "children"),
                  Output("data-output5", "children"), 
                  Output("data-output6", "children"),
                  [Input("interval", "n_intervals")])
    def fetch_data(n):
        response = requests.get("http://127.0.0.1:5000/data") 
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            total_transaction, fraud_cases, fraud_percentage = utils.trasaction_summary(data, include_other=True)
            return html.Div(f"{total_transaction}"), html.Div(f" {fraud_cases}"), html.Div(f"{fraud_percentage} %")
        else:
            return html.Div("Failed to load data")
               

    @app.callback(Output('geographic-plots', 'figure'), Input('interval', 'n_intervals'))
    def geo_plot(_):
        response = requests.get("http://127.0.0.1:5000/data")
        
        if response.status_code == 200:
            data = pd.DataFrame(response.json())

            fraud_country = data.loc[data['class'] == 1].groupby(by='Country').size().reset_index(name='Fraud_Count')
            fraud_country = fraud_country.replace({'United States': 'United States of America'})

            url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
            gdf = gpd.read_file(url)

            merged = gdf.merge(fraud_country, how='left', left_on='SOVEREIGNT', right_on='Country')
            merged['Fraud_Count'] = merged['Fraud_Count'].fillna(0)

            fig = px.choropleth(merged, locations='Country', locationmode='country names', color='Fraud_Count', 
                                title='Geographic Distribution of Fraud Cases', color_continuous_scale='Reds')
            fig.update_layout(geo=dict(showframe=False, projection_type='natural earth'))
            fig.update_geos(showcountries=True, countrycolor="black", showcoastlines=True)

            return fig


        else:
            return html.Div("Failed to load data")     


    @app.callback(Output('browser-fraud', 'figure'), Input('interval', 'n_intervals')) 
    def plot_fraud_bar(_):     
        response = requests.get("http://127.0.0.1:5000/data") 
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            fraud_case = data[data['class'] == 1].groupby(by='browser').size()
            fraud_case = pd.DataFrame(fraud_case).rename({0 : "Count"}, axis=1).sort_values(by='Count')
            fig = px.bar(fraud_case,x=fraud_case.index, y='Count')

            return fig
            
        else:
            return html.Div("Failed to load data")  
        

    @app.callback(Output('timeseries-plots', 'figure'), Input('time-value', 'value')) 
    def plot_date_timeseries(time_interval):     
        response = requests.get("http://127.0.0.1:5000/data") 
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
           
            data['purchase_time'] = pd.to_datetime(data['purchase_time'])
            fraud_data = data[data['class'] == 1]
            
            if time_interval == 'Date':
                fraud_count = fraud_data.groupby(data['purchase_time'].dt.date).size().reset_index(name='Fraud_Count')
            elif time_interval == 'Month':
                fraud_count = fraud_data.groupby(data['purchase_time'].dt.month).size().reset_index(name='Fraud_Count')
            elif time_interval == 'Day':
                fraud_count = fraud_data.groupby(data['purchase_time'].dt.day).size().reset_index(name='Fraud_Count')
            elif time_interval == 'Hour':
                fraud_count = fraud_data.groupby(data['purchase_time'].dt.hour).size().reset_index(name='Fraud_Count')
            elif time_interval  == 'Minutes':
                fraud_count = fraud_data.groupby(data['purchase_time'].dt.minute).size().reset_index(name='Fraud_Count')

            fig = px.line(fraud_count, y='Fraud_Count', x='purchase_time')

            return fig
            
        else:
            return html.Div("Failed to load data")       
        
    @app.callback(Output('device-distribution', 'figure'), Input('row-slider', 'value')) 
    def plot_devide_distribution(row_value):
        response = requests.get("http://127.0.0.1:5000/data") 
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            fraud_case = data[data['class'] == 1].groupby(by='device_id').size()
            fraud_case = pd.DataFrame(fraud_case).rename({0 : "Count"}, axis=1).sort_values(by='Count', ascending=False).head(row_value)
            fig = px.bar(fraud_case,x=fraud_case.index, y='Count')            

            return fig
            
        else:
            return html.Div("Failed to load data")          

    return app
