import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots

import warnings
warnings.simplefilter(action='ignore', category=pd.core.common.SettingWithCopyWarning)

from app_layout import (
    device_layout,
    model_layout,
    model_precision_layout,
    power_budget_layout,
    resource_layout,
    dllce_layout,
    load_data,
    load_resource_data,
    load_dllce
)

from app_callbacks import (
    device_callback,
    model_callback,
    model_precision_callback,
    power_budget_callback,
    resource_callback,
    dllce_callback
)

data_c, region_map, embodied_map = load_data()
res_data_dict = load_resource_data()
dlcce_dict = load_dllce()

# init app
# app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
external_stylesheets = [
    dbc.themes.BOOTSTRAP,
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css'
]
app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Device", href="/device", active="exact")),
        dbc.NavItem(dbc.NavLink("Models", href="/models", active="exact")),
        dbc.NavItem(dbc.NavLink("Precision", href="/model-precision", active="exact")),
        dbc.NavItem(dbc.NavLink("Power", href="/power-budget", active="exact")),
        dbc.NavItem(dbc.NavLink("DLLCE", href="/dllce", active="exact")),
        dbc.NavItem(dbc.NavLink("GPU vs CPU", href="/resources", active="exact")),
        dbc.NavItem(
            dbc.NavLink(
                "Calculator",
                href="/calculator",
                active="exact",
                className='resource-button',
                style={
                    'border': '1px solid #fff',  # white border
                    'padding': '5px 10px',
                    'margin': '0 10px',
                    'borderRadius': '5px',
                    'backgroundColor': '#4e73df', 
                    'color': '#fff',  # white text
                }
            )
        ),
    ],
    brand="Carbon Footprint of DL Inference on Edge",
    brand_href="/device",
    color="primary",
    dark=True,
)

layouts = {
    '/device': device_layout(),
    '/models': model_layout(),
    '/model-precision': model_precision_layout(),
    '/power-budget': power_budget_layout(),
    '/resources': resource_layout(),
    '/dllce': dllce_layout(),
}

callbacks = {
    '/device': device_callback(app, data_c, region_map, embodied_map),
    '/models': model_callback(app, data_c, region_map, embodied_map),
    '/model-precision': model_precision_callback(app, data_c, region_map, embodied_map),
    '/power-budget': power_budget_callback(app, data_c, region_map, embodied_map),
    '/resources': resource_callback(app, res_data_dict),
    '/dllce': dllce_callback(app, dlcce_dict, np.linspace(100, 1000000000, 10000)),
}

# set up app layout
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname in layouts:
        return layouts[pathname]
    else:
        return '404: Not Found'

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    navbar,
    html.Div(id='page-content')
])

# register callback
for path, clbk in callbacks.items():
    clbk


if __name__ == "__main__":
    app.run_server(debug=True)