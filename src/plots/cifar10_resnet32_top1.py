import copy

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from copy import deepcopy


PALETTE = [
    '#a8dadc',
    '#fe5e51',
    '#f6db5f',
    '#ffb554',
    '#36abb5',
    '#808080',
    '#4f5d75',
    '#9e3d64',
]


resnet32_cifar10_data = {
    'Width Multiplier': {
        # Taken from Structured Multi-hashing paper: https://arxiv.org/pdf/1911.11177.pdf
        'baseline': 92.6,
        'data': [
            (2.0, 72.5),
            (6.5, 84.5),
            (14.5, 87.6),
            (25.05, 89.0),
        ],
        'mode': 'lines+markers',
        'marker': {
            'size': 12,
            'symbol': 'cross',
            'color': PALETTE[0]
        },
        'line': {
            'color': PALETTE[0],
            'dash': None,
            'width': 3
        }
    },
    'Wide Compression (TR)': {
        # 1802.09052
        'baseline': 100-7.5,
        'data': [
            (100/115, 100-22.2),
            (100/15, 100-19.2),
            (100/5, 100-9.4),
        ],
        'mode': 'lines+markers',
        'marker': {
            'size': 12,
            'symbol': 'diamond',
            'color': PALETTE[1]
        },
        'line': {
            'color': PALETTE[1],
            'dash': None,
            'width': 3
        }
    },
    'Structured Multi-Hashing': {
        # 1911.11177
        'baseline': 92.6,
        'data': [
            (1.1, 76),
            (1.4, 79.8),
            (2.5, 83),
            (3.0, 86.5),
            (3.5, 86.8),
            (7.0, 88.9),
            (8.0, 89.5),
            (10.1, 90.1),
            (13.0, 90.3),
            (16.3, 92.0),
            (19.5, 91.5),
            (22.5, 91.5),
            (25.5, 92.5),
        ],
        'mode': 'lines+markers',
        'marker': {
            'size': 12,
            'symbol': 'x',
            'color': PALETTE[2]
        },
        'line': {
            'color': PALETTE[2],
            'dash': None,
            'width': 3
        }
    },
    'CircConv': {
        # 1902.11268
        'baseline': 100-7.5,
        'data': [
            (82, 100-7.35),
            (67, 100-7.45),
            (58, 100-7.8),
            (41, 100-8.25),
            (38, 100-8.6),
            (22, 100-9.5),
        ],
        'mode': 'lines+markers',
        'marker': {
            'size': 12,
            'symbol': 'triangle-down',
            'color': PALETTE[3],
        },
        'line': {
            'color': PALETTE[3],
            'dash': None,
            'width': 3
        }
    },
    'Factorized Conv. Filters': {
        # Compressing Convolutional Neural Networks via Factorized Convolutional Filters
        'baseline': 92.43,
        'data': [
            (100-42.71, 92.43-0.25),
            (100-69.46, 92.43-1.69),
        ],
        'mode': 'markers+lines',
        'marker': {
            'size': 12,
            'symbol': 'triangle-up',
            'color': PALETTE[4]
        },
        'line': {
            'color': PALETTE[4],
            'dash': None,
            'width': 3
        }
    },
    'COP': {
        # 1906.10337
        'baseline': 92.64,
        'data': [
            (100-57.5, 91.97),
        ],
        'mode': 'markers',
        'marker': {
            'size': 16,
            'symbol': 'star',
            'color': PALETTE[1]
        },
        'line': {
            'color': PALETTE[1],
            'dash': None,
            'width': 3
        }
    },
}


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = hex_color * 2
    return int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)


def plot_table(dict_data, title, fname, xaxis_title, yaxis_title, baseline_ann=None, baseline_x=0, **kwargs):
    fig = go.Figure()

    for name, data in dict_data.items():
        v = deepcopy(data)
        v['name'] = name
        v['x'] = [a[0] for a in v['data']]
        v['y'] = [a[1] for a in v['data']]
        if len(v['data'][0]) >= 3:
            v['text'] = [a[2] for a in v['data']]
        if len(v['data'][0]) >= 4:
            for i in (3, 2, 1):
                fig.add_trace(go.Scatter(
                    x=v['x'], y=[y+i*s[3] for y, s in zip(v['y'], v['data'])], fill=None, mode='lines',
                    line={'width': 0}, showlegend=False,
                ))
                fig.add_trace(go.Scatter(
                    x=v['x'], y=[y-i*s[3] for y, s in zip(v['y'], v['data'])], fill='tonexty', mode='lines',
                    line={'width': 0}, showlegend=False, fillcolor=f"rgba{(*hex_to_rgb(v['line']['color']), 0.3)}"
                ))
        v.pop('data')
        v.pop('baseline', None)
        fig.add_trace(go.Scatter(**v))

    fig.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        plot_bgcolor='#FFFFFF',
        xaxis_type=kwargs.pop('xaxis_type', 'log'),
        colorway=px.colors.qualitative.Dark24,
        showlegend=True,
        legend=dict(
            orientation='v', x=0.98, y=0.01, xanchor='right', yanchor='bottom',
            bgcolor=None, bordercolor="Black", borderwidth=3, traceorder='reversed',
            font=kwargs.pop('legend_font_dict', None)
        ),
        width=1024,
        height=650,
        font=dict(
            size=24,
            color="#7f7f7f"
        ),
        **kwargs
    )

    if baseline_ann is not None:
        fig.update_layout(annotations=[
            go.layout.Annotation(
                x=baseline_x,
                y=0,
                xref="x",
                yref="y",
                xanchor='left',
                yanchor='bottom',
                text=baseline_ann,
                showarrow=False,
            )
        ])

    fig.update_xaxes(showline=True, gridwidth=3, gridcolor='#F0F0F0', linecolor="black", linewidth=3)
    fig.update_yaxes(showline=True, gridwidth=3, gridcolor='#F0F0F0', linecolor="black", linewidth=3)

    fig.write_image(fname)
    # fig.show()


def process_resnet32_cifar10_top1(flavor):
    assert flavor in ('net_total', 'net_param')

    df = pd.read_csv('cifar10_resnet32_top1_wandb.csv')
    exps = {}
    compression_limit_total = None
    min_perf = np.inf

    for idx, row in df.iterrows():
        top1 = float(row['metrics_best/top1acc'])
        cmp_with_b = float(row[f'compression/compression_{flavor}_with_basis'])
        cmp_wo_b = float(row[f'compression/compression_{flavor}_without_basis'])
        min_perf = min(min_perf, top1)
        is_baseline = 'baseline' in row.Name
        if is_baseline:
            exp_name = 'baseline'
        else:
            if compression_limit_total is None:
                compression_limit_total = float(row['compression/compression_limit_total'])
            b = int(row['basis_size'])
            r = int(row['basis_rank'])
            exp_name = f'B{b} R{r}'
        if exps.get(exp_name) is None:
            exps[exp_name] = {
                'top1': [top1],
                'cmp_with_b': cmp_with_b,
                'cmp_wo_b': cmp_wo_b,
            }
        else:
            if not is_baseline:
                assert exps[exp_name]['cmp_with_b'] == cmp_with_b, \
                    f"{row.Name}: {exps[exp_name]['cmp_with_b']} == {cmp_with_b}"
                assert exps[exp_name]['cmp_wo_b'] == cmp_wo_b, \
                    f"{row.Name}: {exps[exp_name]['cmp_wo_b']} == {cmp_wo_b}"
            exps[exp_name]['top1'].append(top1)

    for exp_name, exp_data in exps.items():
        top1s = np.array(exp_data['top1'])
        exp_data['top1_mean'] = top1s.mean()
        exp_data['top1_std'] = top1s.std()
        exp_data.pop('top1')

    pt_order = [
        'B2 R2',
        'B4 R2',
        'B2 R4',
        'B4 R4',
        'B8 R4',
        'B4 R8',
        'B8 R8',
        'B16 R8',
        'B8 R16',
        'B16 R16',
        'B32 R16',
        'B16 R32',
        'B32 R32',
    ]

    plots = copy.deepcopy(resnet32_cifar10_data)

    plots['T-Basis (with basis)'] = {
        'data': [
            (exps[tag]['cmp_with_b'], exps[tag]['top1_mean'], tag, exps[tag]['top1_std'])
            for tag in pt_order
        ],
        'mode': 'lines+markers+text',
        'marker': {
            'size': 12,
            'color': PALETTE[6]
        },
        'textposition': 'bottom right',
        'textfont': {'size': 14},
        'line': {
            'color': PALETTE[6],
            'dash': None,
            'width': 4
        }
    }

    plots['T-Basis (without basis)'] = {
        'data': [
            (exps[tag]['cmp_wo_b'], exps[tag]['top1_mean'], tag)
            for tag in pt_order
        ],
        'mode': 'lines+markers+text',
        'marker': {
            'size': 12,
            'color': PALETTE[7]
        },
        'textposition': 'top left',
        'textfont': {'size': 14},
        'line': {
            'color': PALETTE[7],
            'dash': None,
            'width': 4
        }
    }

    plots['Baseline'] = {
        'data': [
            (0, exps['baseline']['top1_mean']),
            (100, exps['baseline']['top1_mean']),
        ],
        'mode': 'lines',
        'marker': {
            'color': 'black'
        },
        'line': {
            'color': 'black',
            'dash': 'dash',
            'width': 3
        }
    }

    if flavor == 'net_total':
        plots['Compression Limit'] = {
            'data': [
                (compression_limit_total, min_perf),
                (compression_limit_total, exps['baseline']['top1_mean']),
            ],
            'mode': 'lines',
            'marker': {
                'color': 'black'
            },
            'line': {
                'color': 'black',
                'dash': 'dot',
                'width': 3
            }
        }

    title_suffix = {
        'net_total': '(parameters and buffers)',
        'net_param': '(parameters only)',
    }[flavor]

    plot_table(
        plots,
        title=f'ResNet32-CIFAR10-Top1 {title_suffix}',
        fname=f'resnet32_cifar10_top1_{flavor}.pdf',
        xaxis_title='Compression %',
        yaxis_title='Accuracy %',
        baseline_ann=None,
        legend_font_dict=dict(size=22)
    )


if __name__ == '__main__':
    process_resnet32_cifar10_top1('net_param')
    process_resnet32_cifar10_top1('net_total')
