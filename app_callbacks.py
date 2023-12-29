from dash.dependencies import Input, Output
from app_layout import load_data
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots

def metrics(data, n_inferences, region_conv, embodied_map):
    data.loc[:, 'total_infer_time'] = n_inferences / data['mean_throughput']
    data.loc[:, 'power_per_image'] = data["mean_power"] / data["batch_size"]
    data.loc[:, 'total_power'] = data['power_per_image'] * n_inferences
    data.loc[:, 'total_energy'] = data['total_power'] * data['total_infer_time']
    data.loc[:, 'total_oper_carbon'] = (data['total_energy'] / (3.6 * (10**9))) * region_conv
    data.loc[:, 'total_oper_carbon'] = data['total_oper_carbon'] * 1e3  # in grams

    lifetime = 5  # in years
    data.loc[:, 'emb_t_ratio'] = data['total_infer_time'] / (lifetime * 365 * 86400)
    data.loc[:, 'total_emb_carbon'] = data['emb_t_ratio'] * data['hw_name'].map(embodied_map)
    data.loc[:, 'total_emb_carbon'] = data['total_emb_carbon'] * 1e6  # in mg
    return data

def plot_dual(x, y_prim, y_sec, name_prim, name_sec, mode, text_prim, text_sec, title_text, xaxis, yaxis_prim, yaxis_sec):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=x,
        y=y_sec,
        name=name_sec,
        mode=mode,
        text=text_sec,
        textposition='bottom center',
        texttemplate='%{text:.1f}'
    ), secondary_y=True)
    fig.add_trace(go.Bar(
        x=x,
        y=y_prim,
        name=name_prim,
        text=text_prim,
        textposition='outside',
        texttemplate='%{text:.1f}'
    ), secondary_y=False)
    fig.update_layout(
        title_text=title_text,
        title_x=0.5,
        template='plotly_white',
        xaxis=xaxis,
        yaxis=yaxis_prim,
        yaxis2=yaxis_sec,
        autosize=True,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_single(data, x, y, color, barmode, text, title_text, xaxis, yaxis, legend):
    fig = px.bar(
        data, 
        x=x, 
        y=y, 
        color=color,
        barmode=barmode,
        text=text
    )
    fig.update_traces(textposition='outside', texttemplate='%{text:.1f}')
    fig.update_layout(
        title_text=title_text,
        title_x=0.5,
        template='plotly_white',
        xaxis=xaxis,
        yaxis=yaxis,
        legend=legend,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig 

def device_callback(app, data_c, region_map, embodied_map):
    @app.callback(
        [
            Output('device_graph1', 'figure'),
            Output('device_graph2', 'figure'),
            Output('device_graph3', 'figure'),
            Output('device_graph4', 'figure')
        ],
        [
            Input('device_hw_name', 'value'),
            Input('device_model_name', 'value'),
            Input('device_precision', 'value'),
            Input('device_power_budget', 'value'),
            Input('n_inferences', 'value'),
            Input('region', 'value'),
            Input('batch_size', 'value')
        ]
    )
    def update_device_graphs(hw_names, model_name, precision, power_budget, n_inferences, region, batch_size):
        region_conv = region/2205.0
        if 'nano8g' in hw_names:
            nano_data = data_c[
                (data_c['hw_name'] == 'nano8g') &
                (data_c['model_name'] == model_name) &
                (data_c['precision'] == precision) &
                (data_c['batch_size'] == batch_size)
            ]
        else:
            nano_data = pd.DataFrame()
        data_ = data_c[
                (data_c['hw_name'].isin(hw_names)) &
                (data_c['model_name'] == model_name) &
                (data_c['precision'] == precision) &
                (data_c['power_mode'] == power_budget) & 
                (data_c['batch_size'] == batch_size)
            ]
        data_ = pd.concat([data_, nano_data], ignore_index=True)
        
        filt_data = metrics(data_, n_inferences, region_conv, embodied_map)

        # operational and embodied carbon
        fig_1 = plot_dual(
            x=filt_data['hw_name'],
            y_prim=filt_data['total_oper_carbon'],
            y_sec=filt_data['total_emb_carbon'],
            name_prim='Operational Carbon',
            name_sec='Embodied Carbon',
            mode='lines+markers+text',
            text_prim=filt_data['total_oper_carbon'].round(1),
            text_sec=filt_data['total_emb_carbon'].round(1),
            title_text='Operational and Embodied Carbon',
            xaxis=dict(title='Device', showgrid=False),
            yaxis_prim=dict(title='Operational Carbon (g)', showgrid=False, zeroline=True, showline=True),
            yaxis_sec=dict(title='Embodied Carbon (mg)', showgrid=False, zeroline=True, showline=True)        
        )

        # throughput and throughput per joule
        fig_2 = plot_dual(
            x = filt_data['hw_name'],
            y_prim = filt_data['mean_throughput'],
            y_sec = filt_data['throughput_per_energy(t_mJ)']*1000,
            name_prim = 'Mean Throughput',
            name_sec = 'Throughput per Joule',
            mode = 'lines+markers+text',
            text_prim = filt_data['mean_throughput'].round(1),
            text_sec = (filt_data['throughput_per_energy(t_mJ)']*1000).round(1),
            title_text = 'Throughput and Throughput per Joule',
            xaxis = dict(title='Device', showgrid=False),
            yaxis_prim = dict(title='Throughput (X)', showgrid=False, zeroline=True, showline=True),
            yaxis_sec = dict(title='Throughput per Joule (X / J)', showgrid=False, zeroline=True, showline=True),
        )
        # print(fig_2)

        # data for fig 3 and fig 4
        if 'nano8g' in hw_names:
            nano_data_1 = data_c[
                (data_c['hw_name'] == 'nano8g') &
                (data_c['model_name'] == model_name) &
                (data_c['precision'] == precision)
            ]
        else:
            nano_data_1 = pd.DataFrame()
        data_1_ = data_c[
                (data_c['hw_name'].isin(hw_names)) &
                (data_c['model_name'] == model_name) &
                (data_c['precision'] == precision) &
                (data_c['power_mode'] == power_budget)
            ]
        data_1_ = pd.concat([data_1_, nano_data_1], ignore_index=True)
        filt_data_1 = metrics(data_1_, n_inferences, region_conv, embodied_map)

        # fig 3 inference time vs batch size
        fig_3 = plot_single(
            filt_data_1, 
            x='batch_size', 
            y='total_infer_time', 
            color='hw_name',
            barmode='group',
            text='total_infer_time',
            title_text='Mean Inference Time vs. Batch Size',
            xaxis=dict(title='Batch Size', showgrid=False, type='category'),  # Set x-axis as categorical
            yaxis=dict(title='Total Inference Time (min)', showgrid=False, zeroline=True, showline=True),
            legend=dict(title='Device')
        )

        # fig 4: power consumed per image vs batch size
        fig_4 = plot_single(
            filt_data_1, 
            x='batch_size', 
            y='power_per_image', 
            color='hw_name',
            barmode='group',
            text='power_per_image',
            title_text='Power Consumed Per Image vs. Batch Size',
            xaxis=dict(title='Batch Size', showgrid=False, type='category'),  # Set x-axis as categorical
            yaxis=dict(title='Mean Power per Image (W)', showgrid=False, zeroline=True, showline=True),
            legend=dict(title='Device')
        )

        return fig_1, fig_2, fig_3, fig_4


def model_callback(app, data_c, region_map, embodied_map):
    @app.callback(
        [
            Output('model_graph1', 'figure'),
            Output('model_graph2', 'figure'),
            Output('model_graph3', 'figure'),
            Output('model_graph4', 'figure')
        ],
        [
            Input('model_hw_name', 'value'),
            Input('model_model_name', 'value'),
            Input('model_precision', 'value'),
            Input('model_power_budget', 'value'),
            Input('n_inferences', 'value'),
            Input('region', 'value'),
            Input('batch_size', 'value')
        ]
    )
    def update_model_graphs(hw_name, model_names, precision, power_budget, n_inferences, region, batch_size):
        region_conv = region/2205.0
        if hw_name == 'nano8g':
            data_ = data_c[
                (data_c['hw_name'] == 'nano8g') &
                (data_c['model_name'].isin(model_names)) &
                (data_c['precision'] == precision) &
                (data_c['batch_size'] == batch_size)
            ]
        else:
            data_ = data_c[
                    (data_c['hw_name'] == hw_name) &
                    (data_c['model_name'].isin(model_names)) &
                    (data_c['precision'] == precision) &
                    (data_c['power_mode'] == power_budget) & 
                    (data_c['batch_size'] == batch_size)
                ]
        filt_data = metrics(data_, n_inferences, region_conv, embodied_map)

        # operational and embodied carbon
        fig_1 = plot_dual(
            x = filt_data['model_name'],
            y_prim = filt_data['total_oper_carbon'],
            y_sec = filt_data['total_emb_carbon'],
            name_prim='Operational Carbon',
            name_sec='Embodied Carbon',
            mode='lines+markers+text',
            text_prim=filt_data['total_oper_carbon'].round(1),
            text_sec=filt_data['total_emb_carbon'].round(1),
            title_text='Operational and Embodied Carbon',
            xaxis=dict(title='Model Name', showgrid=False),
            yaxis_prim=dict(title='Operational Carbon (g)', showgrid=False, zeroline=True, showline=True),
            yaxis_sec=dict(title='Embodied Carbon (mg)', showgrid=False, zeroline=True, showline=True)        
        )

        # throughput and throughput per joule
        fig_2 = plot_dual(
            x = filt_data['model_name'],
            y_prim = filt_data['mean_throughput'],
            y_sec = filt_data['throughput_per_energy(t_mJ)']*1000,
            name_prim = 'Mean Throughput',
            name_sec = 'Throughput per Joule',
            mode = 'lines+markers+text',
            text_prim = filt_data['mean_throughput'].round(1),
            text_sec = (filt_data['throughput_per_energy(t_mJ)']*1000).round(1),
            title_text = 'Throughput and Throughput per Joule',
            xaxis=dict(title='Model Name', showgrid=False),
            yaxis_prim = dict(title='Throughput (X)', showgrid=False, zeroline=True, showline=True),
            yaxis_sec = dict(title='Throughput per Joule (X / J)', showgrid=False, zeroline=True, showline=True),
        )

        # data for fig 3 and fig 4
        if hw_name == 'nano8g':
            data_1_ = data_c[
                (data_c['hw_name'] == 'nano8g') &
                (data_c['model_name'].isin(model_names)) &
                (data_c['precision'] == precision)
            ]
        else:
            data_1_ = data_c[
                    (data_c['hw_name'] == hw_name) &
                    (data_c['model_name'].isin(model_names)) &
                    (data_c['precision'] == precision) &
                    (data_c['power_mode'] == power_budget)
                ]
        filt_data_1 = metrics(data_1_, n_inferences, region_conv, embodied_map)

        # fig 3 inference time vs batch size
        fig_3 = plot_single(
            filt_data_1, 
            x='batch_size', 
            y='total_infer_time', 
            color='model_name',
            barmode='group',
            text='total_infer_time',
            title_text='Mean Inference Time vs. Batch Size',
            xaxis=dict(title='Batch Size', showgrid=False, type='category'),  # Set x-axis as categorical
            yaxis=dict(title='Total Inference Time (min)', showgrid=False, zeroline=True, showline=True),
            legend=dict(title='Model')
        )

        # fig 4: power consumed per image vs batch size
        fig_4 = plot_single(
            filt_data_1, 
            x='batch_size', 
            y='power_per_image', 
            color='model_name',
            barmode='group',
            text='power_per_image',
            title_text='Power Consumed Per Image vs. Batch Size',
            xaxis=dict(title='Batch Size', showgrid=False, type='category'),  # Set x-axis as categorical
            yaxis=dict(title='Mean Power per Image (W)', showgrid=False, zeroline=True, showline=True),
            legend=dict(title='Model')
        )

        return fig_1, fig_2, fig_3, fig_4

def model_precision_callback(app, data_c, region_map, embodied_map):
    @app.callback(
        [
            Output('model_precision_graph1', 'figure'),
            Output('model_precision_graph2', 'figure'),
            Output('model_precision_graph3', 'figure'),
            Output('model_precision_graph4', 'figure')
        ],
        [
            Input('model_precision_hw_name', 'value'),
            Input('model_precision_model_name', 'value'),
            Input('model_precision_precision', 'value'),
            Input('model_precision_power_budget', 'value'),
            Input('n_inferences', 'value'),
            Input('region', 'value'),
            Input('batch_size', 'value')
        ]
    )
    def update_model_precision_graphs(hw_name, model_name, precision_list, power_budget, n_inferences, region, batch_size):
        region_conv = region/2205.0
        if hw_name == 'nano8g':
            data_ = data_c[
                (data_c['hw_name'] == 'nano8g') &
                (data_c['model_name'] == model_name) &
                (data_c['precision'].isin(precision_list)) &
                (data_c['batch_size'] == batch_size)
            ]
        else:
            data_ = data_c[
                    (data_c['hw_name'] == hw_name) &
                    (data_c['model_name'] == model_name) &
                    (data_c['precision'].isin(precision_list)) &
                    (data_c['power_mode'] == power_budget) & 
                    (data_c['batch_size'] == batch_size)
                ]
        filt_data = metrics(data_, n_inferences, region_conv, embodied_map)

        # operational and embodied carbon
        fig_1 = plot_dual(
            x = filt_data['precision'],
            y_prim = filt_data['total_oper_carbon'],
            y_sec = filt_data['total_emb_carbon'],
            name_prim='Operational Carbon',
            name_sec='Embodied Carbon',
            mode='lines+markers+text',
            text_prim=filt_data['total_oper_carbon'].round(1),
            text_sec=filt_data['total_emb_carbon'].round(1),
            title_text='Operational and Embodied Carbon',
            xaxis=dict(title='Model Precision', showgrid=False),
            yaxis_prim=dict(title='Operational Carbon (g)', showgrid=False, zeroline=True, showline=True),
            yaxis_sec=dict(title='Embodied Carbon (mg)', showgrid=False, zeroline=True, showline=True)        
        )

        # throughput and throughput per joule
        fig_2 = plot_dual(
            x = filt_data['precision'],
            y_prim = filt_data['mean_throughput'],
            y_sec = filt_data['throughput_per_energy(t_mJ)']*1000,
            name_prim = 'Mean Throughput',
            name_sec = 'Throughput per Joule',
            mode = 'lines+markers+text',
            text_prim = filt_data['mean_throughput'].round(1),
            text_sec = (filt_data['throughput_per_energy(t_mJ)']*1000).round(1),
            title_text = 'Throughput and Throughput per Joule',
            xaxis=dict(title='Model Precision', showgrid=False),
            yaxis_prim = dict(title='Throughput (X)', showgrid=False, zeroline=True, showline=True),
            yaxis_sec = dict(title='Throughput per Joule (X / J)', showgrid=False, zeroline=True, showline=True),
        )

        # data for fig 3 and fig 4
        if hw_name == 'nano8g':
            data_1_ = data_c[
                (data_c['hw_name'] == 'nano8g') &
                (data_c['model_name'] == model_name) &
                (data_c['precision'].isin(precision_list))
            ]
        else:
            data_1_ = data_c[
                    (data_c['hw_name'] == hw_name) &
                    (data_c['model_name'] == model_name) &
                    (data_c['precision'].isin(precision_list)) & 
                    (data_c['power_mode'] == power_budget)
                ]
        filt_data_1 = metrics(data_1_, n_inferences, region_conv, embodied_map)

        # fig 3 inference time vs batch size
        fig_3 = plot_single(
            filt_data_1, 
            x='batch_size', 
            y='total_infer_time', 
            color='precision',
            barmode='group',
            text='total_infer_time',
            title_text='Mean Inference Time vs. Batch Size',
            xaxis=dict(title='Batch Size', showgrid=False, type='category'),  # Set x-axis as categorical
            yaxis=dict(title='Total Inference Time (min)', showgrid=False, zeroline=True, showline=True),
            legend=dict(title='Precision')
        )

        # fig 4: power consumed per image vs batch size
        fig_4 = plot_single(
            filt_data_1, 
            x='batch_size', 
            y='power_per_image', 
            color='precision',
            barmode='group',
            text='power_per_image',
            title_text='Power Consumed Per Image vs. Batch Size',
            xaxis=dict(title='Batch Size', showgrid=False, type='category'),  # Set x-axis as categorical
            yaxis=dict(title='Mean Power per Image (W)', showgrid=False, zeroline=True, showline=True),
            legend=dict(title='Precision')
        )

        return fig_1, fig_2, fig_3, fig_4

def power_budget_callback(app, data_c, region_map, embodied_map):
    @app.callback(
        [
            Output('power_budget_graph1', 'figure'),
            Output('power_budget_graph2', 'figure'),
            Output('power_budget_graph3', 'figure'),
            Output('power_budget_graph4', 'figure')
        ],
        [
            Input('power_budget_hw_name', 'value'),
            Input('power_budget_model_name', 'value'),
            Input('power_budget_precision', 'value'),
            Input('power_budget_power_budget', 'value'),
            Input('n_inferences', 'value'),
            Input('region', 'value'),
            Input('batch_size', 'value')
        ]
    )
    def update_power_budget_graphs(hw_name, model_name, precision, power_budget_list, n_inferences, region, batch_size):
        region_conv = region/2205.0
        if hw_name == 'nano8g':
            data_ = data_c[
                (data_c['hw_name'] == 'nano8g') &
                (data_c['model_name'] == model_name) &
                (data_c['precision'] == precision) &
                (data_c['batch_size'] == batch_size)
            ]
        else:
            data_ = data_c[
                    (data_c['hw_name'] == hw_name) &
                    (data_c['model_name'] == model_name) &
                    (data_c['precision'] == precision) &
                    (data_c['power_mode'].isin(power_budget_list)) & 
                    (data_c['batch_size'] == batch_size)
                ]
        filt_data = metrics(data_, n_inferences, region_conv, embodied_map)

        # operational and embodied carbon
        fig_1 = plot_dual(
            x = filt_data['power_mode'],
            y_prim = filt_data['total_oper_carbon'],
            y_sec = filt_data['total_emb_carbon'],
            name_prim='Operational Carbon',
            name_sec='Embodied Carbon',
            mode='lines+markers+text',
            text_prim=filt_data['total_oper_carbon'].round(1),
            text_sec=filt_data['total_emb_carbon'].round(1),
            title_text='Operational and Embodied Carbon',
            xaxis=dict(title='Power Budget', showgrid=False),
            yaxis_prim=dict(title='Operational Carbon (g)', showgrid=False, zeroline=True, showline=True),
            yaxis_sec=dict(title='Embodied Carbon (mg)', showgrid=False, zeroline=True, showline=True)        
        )

        # throughput and throughput per joule
        fig_2 = plot_dual(
            x = filt_data['power_mode'],
            y_prim = filt_data['mean_throughput'],
            y_sec = filt_data['throughput_per_energy(t_mJ)']*1000,
            name_prim = 'Mean Throughput',
            name_sec = 'Throughput per Joule',
            mode = 'lines+markers+text',
            text_prim = filt_data['mean_throughput'].round(1),
            text_sec = (filt_data['throughput_per_energy(t_mJ)']*1000).round(1),
            title_text = 'Throughput and Throughput per Joule',
            xaxis=dict(title='Power Budget', showgrid=False),
            yaxis_prim = dict(title='Throughput (X)', showgrid=False, zeroline=True, showline=True),
            yaxis_sec = dict(title='Throughput per Joule (X / J)', showgrid=False, zeroline=True, showline=True),
        )

        # data for fig 3 and fig 4
        if hw_name == 'nano8g':
            data_1_ = data_c[
                (data_c['hw_name'] == 'nano8g') &
                (data_c['model_name'] == model_name) &
                (data_c['precision'] == precision)
            ]
        else:
            data_1_ = data_c[
                    (data_c['hw_name'] == hw_name) &
                    (data_c['model_name'] == model_name) &
                    (data_c['precision'] == precision) & 
                    (data_c['power_mode'].isin(power_budget_list))
                ]
        filt_data_1 = metrics(data_1_, n_inferences, region_conv, embodied_map)

        # fig 3 inference time vs batch size
        fig_3 = plot_single(
            filt_data_1, 
            x='batch_size', 
            y='total_infer_time', 
            color='power_mode',
            barmode='group',
            text='total_infer_time',
            title_text='Mean Inference Time vs. Batch Size',
            xaxis=dict(title='Batch Size', showgrid=False, type='category'),  # Set x-axis as categorical
            yaxis=dict(title='Total Inference Time (min)', showgrid=False, zeroline=True, showline=True),
            legend=dict(title='Power Budget')
        )

        # fig 4: power consumed per image vs batch size
        fig_4 = plot_single(
            filt_data_1, 
            x='batch_size', 
            y='power_per_image', 
            color='power_mode',
            barmode='group',
            text='power_per_image',
            title_text='Power Consumed Per Image vs. Batch Size',
            xaxis=dict(title='Batch Size', showgrid=False, type='category'),  # Set x-axis as categorical
            yaxis=dict(title='Mean Power per Image (W)', showgrid=False, zeroline=True, showline=True),
            legend=dict(title='Power Budget')
        )

        return fig_1, fig_2, fig_3, fig_4
