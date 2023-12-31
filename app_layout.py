from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd 

def load_data():
    all_data = pd.read_csv('all_data.csv')
    filt_cols = [col for col in all_data.columns if not col.startswith('std_')]
    data_c = all_data[filt_cols].copy()

    region_data = pd.read_csv("region_co2.csv")
    emb_data = pd.read_csv("embodied_co2.csv")
    region_map = dict(zip(region_data['Region'], region_data['CO2_Intensity']))
    embodied_map = dict(zip(emb_data['Device'], emb_data['Embodied_Carbon']))

    return data_c, region_map, embodied_map

def load_resource_data():
    mem = pd.read_csv('all_mem.csv')
    cpu = pd.read_csv('all_cpu.csv')
    pow = pd.read_csv('all_pow.csv')
    cpu_temp = pd.read_csv('all_cpu_temp.csv')
    res_dict = {
        "all_mem": mem,
        "all_mem_util": mem,
        "all_cpu": cpu,
        "all_power": pow,
        "all_cpu_temp": cpu_temp
    }
    return res_dict

def base_layout(region_map, b_size):
    layout = dbc.Container([
        html.Hr(),
        
        dbc.Row([
            dbc.Col([
                html.Label('Number of Inferences'),
                dcc.Input(
                    id='n_inferences',
                    type='number',
                    value=8392,
                    style={'width': '100%', 'display': 'inline-block'}
                ),
                html.Div(id='n_inferences_output')
            ], width=4),
            dbc.Col([
                html.Label('Region'),
                dcc.Dropdown(
                    id='region',
                    options=[{'label': k, 'value': v} for k, v in region_map.items()],
                    value=region_map['Indiana']
                ),
            ], width=4),
            dbc.Col([
                html.Label('Batch Size'),
                dcc.Dropdown(
                    id='batch_size',
                    options=[{'label': b, 'value': b} for b in b_size],
                    value=1
                ),
            ], width=4)
        ]),

        html.Hr(),
    ])
    return layout    

# device page
def device_layout():
    data_c, region_map, embodied_map = load_data()
    b_size = data_c['batch_size'].unique()
    models = data_c['model_name'].unique()
    model_precision = data_c['precision'].unique()
    power_budget = data_c['power_mode'].unique()
    hw_name = data_c['hw_name'].unique()

    device_graph_row_1 = dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='device_graph1'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='device_graph2'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
    ], style={'margin-top': '20px'})

    device_graph_row_2 = dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='device_graph3'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='device_graph4'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
    ], style={'margin-top': '20px'})

    layout = base_layout(region_map, b_size)
    layout.children += ([
        dbc.Row([
        dbc.Col([
            html.H3("Device", style={'margin-top': '5px'})
        ], width=8)
        ]),

        dbc.Row([
            dbc.Col([
                html.Label('Device'),
                dcc.Dropdown(
                    id='device_hw_name',
                    options=[{'label': hw, 'value': hw} for hw in hw_name],
                    multi=True,
                    value=[hw_name[0], hw_name[1]]
                ),
            ], width=3),
            dbc.Col([
                html.Label('Model'),
                dcc.Dropdown(
                    id='device_model_name',
                    options=[{'label': m, 'value': m} for m in models],
                    value=models[0]
                ),
            ], width=3),
            dbc.Col([
                html.Label('Precision'),
                dcc.Dropdown(
                    id='device_precision',
                    options=[{'label': p, 'value': p} for p in model_precision],
                    value='fp32'
                ),
            ], width=3),
            dbc.Col([
                html.Label('Power Budget'),
                dcc.Dropdown(
                    id='device_power_budget',
                    options=[{'label': pb, 'value': pb} for pb in power_budget],
                    value='50W'
                ),
            ], width=3)
        ]),

        html.Hr(),
        device_graph_row_1,
        device_graph_row_2,
        html.Hr(),
    ])
    return layout

# models page
def model_layout():
    data_c, region_map, embodied_map = load_data()
    b_size = data_c['batch_size'].unique()
    models = data_c['model_name'].unique()
    model_precision = data_c['precision'].unique()
    power_budget = data_c['power_mode'].unique()
    hw_name = data_c['hw_name'].unique()

    model_graph_row_1 = dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='model_graph1'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='model_graph2'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
    ], style={'margin-top': '20px'})

    model_graph_row_2 = dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='model_graph3'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='model_graph4'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
    ], style={'margin-top': '20px'})

    layout = base_layout(region_map, b_size)
    layout.children += ([
        dbc.Row([
        dbc.Col([
            html.H3("Models", style={'margin-top': '5px'})
        ], width=8)
        ]),

        dbc.Row([
            dbc.Col([
                html.Label('Device'),
                dcc.Dropdown(
                    id='model_hw_name',
                    options=[{'label': hw, 'value': hw} for hw in hw_name],
                    value=hw_name[0]
                ),
            ], width=3),
            dbc.Col([
                html.Label('Model'),
                dcc.Dropdown(
                    id='model_model_name',
                    options=[{'label': m, 'value': m} for m in models],
                    value=list(models),
                    multi=True
                ),
            ], width=3),
            dbc.Col([
                html.Label('Precision'),
                dcc.Dropdown(
                    id='model_precision',
                    options=[{'label': p, 'value': p} for p in model_precision],
                    value='fp32'
                ),
            ], width=3),
            dbc.Col([
                html.Label('Power Budget'),
                dcc.Dropdown(
                    id='model_power_budget',
                    options=[{'label': pb, 'value': pb} for pb in power_budget],
                    value='50W'
                ),
            ], width=3)
        ]),    

        html.Hr(),
        model_graph_row_1,
        model_graph_row_2,
        html.Hr(),
    ])
    return layout


# model precision page
def model_precision_layout():
    data_c, region_map, embodied_map = load_data()
    b_size = data_c['batch_size'].unique()
    models = data_c['model_name'].unique()
    model_precision = data_c['precision'].unique()
    power_budget = data_c['power_mode'].unique()
    hw_name = data_c['hw_name'].unique()

    model_precision_graph_row_1 = dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='model_precision_graph1'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='model_precision_graph2'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
    ], style={'margin-top': '20px'})

    model_precision_graph_row_2 = dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='model_precision_graph3'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='model_precision_graph4'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
    ], style={'margin-top': '20px'})

    layout = base_layout(region_map, b_size)
    layout.children += ([
        dbc.Row([
        dbc.Col([
            html.H3("Model Precision", style={'margin-top': '5px'})
        ], width=8)
        ]),

        dbc.Row([
            dbc.Col([
                html.Label('Device'),
                dcc.Dropdown(
                    id='model_precision_hw_name',
                    options=[{'label': hw, 'value': hw} for hw in hw_name],
                    value=hw_name[0]
                ),
            ], width=3),
            dbc.Col([
                html.Label('Model'),
                dcc.Dropdown(
                    id='model_precision_model_name',
                    options=[{'label': m, 'value': m} for m in models],
                    value=models[0]
                ),
            ], width=3),
            dbc.Col([
                html.Label('Precision'),
                dcc.Dropdown(
                    id='model_precision_precision',
                    options=[{'label': p, 'value': p} for p in model_precision],
                    multi=True,
                    value=list(model_precision)
                ),
            ], width=3),
            dbc.Col([
                html.Label('Power Budget'),
                dcc.Dropdown(
                    id='model_precision_power_budget',
                    options=[{'label': pb, 'value': pb} for pb in power_budget],
                    value='50W'
                ),
            ], width=3)
        ]),  

        html.Hr(),
        model_precision_graph_row_1,
        model_precision_graph_row_2,
        html.Hr(),
    ])
    return layout


# power budget page
def power_budget_layout():
    data_c, region_map, embodied_map = load_data()
    b_size = data_c['batch_size'].unique()
    models = data_c['model_name'].unique()
    model_precision = data_c['precision'].unique()
    power_budget = data_c['power_mode'].unique()
    hw_name = data_c['hw_name'].unique()

    power_budget_graph_row_1 = dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='power_budget_graph1'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='power_budget_graph2'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
    ], style={'margin-top': '20px'})

    power_budget_graph_row_2 = dbc.Row([
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='power_budget_graph3'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
        dbc.Col(
            dbc.Card(
                dcc.Graph(id='power_budget_graph4'),
                body=True,
                className="border-grey mb-3",
                style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
            ),
            width=6
        ),
    ], style={'margin-top': '20px'})

    layout = base_layout(region_map, b_size)
    layout.children += ([
        dbc.Row([
        dbc.Col([
            html.H3("Power Budget", style={'margin-top': '5px'})
        ], width=8)
        ]),

        dbc.Row([
            dbc.Col([
                html.Label('Device'),
                dcc.Dropdown(
                    id='power_budget_hw_name',
                    options=[{'label': hw, 'value': hw} for hw in hw_name],
                    value=hw_name[0]
                ),
            ], width=3),
            dbc.Col([
                html.Label('Model'),
                dcc.Dropdown(
                    id='power_budget_model_name',
                    options=[{'label': m, 'value': m} for m in models],
                    value=models[0]
                ),
            ], width=3),
            dbc.Col([
                html.Label('Precision'),
                dcc.Dropdown(
                    id='power_budget_precision',
                    options=[{'label': p, 'value': p} for p in model_precision],
                    value='fp32'
                ),
            ], width=3),
            dbc.Col([
                html.Label('Power Budget'),
                dcc.Dropdown(
                    id='power_budget_power_budget',
                    options=[{'label': pb, 'value': pb} for pb in power_budget],
                    multi=True,
                    value=list(power_budget)
                ),
            ], width=3)
        ]),
        html.Hr(),
        power_budget_graph_row_1,
        power_budget_graph_row_2,
        html.Hr(),
    ])
    return layout

# resources page
def resource_layout():
    data_c, region_map, embodied_map = load_data()
    b_size = data_c['batch_size'].unique()
    model_precision = data_c['precision'].unique()
    power_budget = data_c['power_mode'].unique()
    model_list = data_c['model_name'].unique()
    res_label_map = {
        'Memory Usage (MB)': 'all_mem',
        'Memory Util (%)': 'all_mem_util',
        'CPU Util (%)': 'all_cpu',
        'Power (mW)': 'all_power',
        'CPU Temp (C)': 'all_cpu_temp',
        'Throughput': 'all_throughput'
    }
    gpu_map = {"Nvidia AGX64GB": 'agx64g', "Nvidia Nano8GB": 'nano8g'}

    layout = dbc.Container([
        html.Hr(),
        html.Div(className="row", children=[
            html.Div(className='six columns', children=[
                html.Label('GPU Device'),
                dcc.Dropdown(
                    id='res_gpu',
                    options=[{'label': k, 'value': v} for k, v in gpu_map.items()],
                    value='agx64g'
                )
            ], style=dict(width='20%')),

            html.Div(className='six columns', children=[
                html.Label('Batch Size'),
                dcc.Dropdown(
                    id='res_batch_size',
                    options=[{'label': b, 'value': b} for b in b_size],
                    value=1
                )
            ], style=dict(width='20%')),

            html.Div(className='six columns', children=[
                html.Label('Precision'),
                dcc.Dropdown(
                    id='res_precision',
                    options=[{'label': p, 'value': p} for p in model_precision],
                    value='fp32'
                )
            ], style=dict(width='20%')),

            html.Div(className='six columns', children=[
                html.Label('Power Budget'),
                dcc.Dropdown(
                    id='res_power_budget',
                    options=[{'label': pb, 'value': pb} for pb in power_budget],
                    value=power_budget[0]
                )
            ], style=dict(width='20%')),

            html.Div(className='six columns', children=[
                html.Label('Metric'),
                dcc.Dropdown(
                    id='res_metric',
                    options=[{'label': k, 'value': res_label_map[k]} for k in res_label_map],
                    value='all_mem'
                )
            ], style=dict(width='20%')),
        ], style=dict(display='flex', justifyContent='space-between')),
        
    html.Div(className='twelve columns', children=[
        dbc.Row([
            dbc.Col([
                dbc.Card(
                    dcc.Graph(id='resource_graph1'),
                    body=True,
                    className="border-grey mb-3",
                    style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
                )
            ], width=6),
            dbc.Col([
                dbc.Card(
                    dcc.Graph(id='resource_graph2'),
                    body=True,
                    className="border-grey mb-3",
                    style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
                )
            ], width=6),
        ], style={'margin-top': '20px'}),
        dbc.Row([
            dbc.Col([
                html.Label('Model'),
                dcc.Dropdown(
                    id='res_models',
                    options=[{'label': model, 'value': model} for model in model_list],
                    value=model_list[0]  
                )
            ], width=12, style={'margin-top': '10px'}),
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dcc.Graph(id='resource_graph3'),
                    body=True,
                    className="border-grey mb-3",
                    style={'border': '1px solid rgba(169, 169, 169, 0.5)', 'border-radius': '5px', 'padding': '10px'}
                ), 
                width=12
            ),
        ]),
    ]),

    html.Hr(),
    ])

    return layout
