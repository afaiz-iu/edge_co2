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
    load_data
)

from app_callbacks import (
    device_callback,
    model_callback,
    model_precision_callback,
    power_budget_callback
)

data_c, region_map, embodied_map = load_data()

# init app
# app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Device", href="/device", active="exact")),
        dbc.NavItem(dbc.NavLink("Models", href="/models", active="exact")),
        dbc.NavItem(dbc.NavLink("Model Precision", href="/model-precision", active="exact")),
        dbc.NavItem(dbc.NavLink("Power Budget", href="/power-budget", active="exact")),
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
}

callbacks = {
    '/device': device_callback(app, data_c, region_map, embodied_map),
    '/models': model_callback(app, data_c, region_map, embodied_map),
    '/model-precision': model_precision_callback(app, data_c, region_map, embodied_map),
    '/power-budget': power_budget_callback(app, data_c, region_map, embodied_map),
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