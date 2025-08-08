import altair as alt

from torch import Tensor
from pandas import DataFrame
from typing import Union
from numpy import ndarray, where


def plot_relative_dist(
    metric_name: str,
    metric: Union[list, ndarray, Tensor],
    labels: Union[list, ndarray, Tensor],
    bin_step: float = 1e-2,
    width: int = 600,
    height: int = 400,
    legend_x: int = 300,
    legend_y: int = 450,
) -> alt.Chart:
    
    data = DataFrame({'score': metric, 'label': labels})

    num_success = len(where(labels == 0)[0])
    num_errors = len(where(labels == 1)[0])

    chart = alt.Chart(data).transform_bin(
        'bin', 
        'score', 
        bin=alt.Bin(step=bin_step, extent=[0, 1])
    ).transform_aggregate(
        count='count()',
        groupby=['bin', 'label']
    ).transform_calculate(
        density=f'datum.count / ((datum.label === 0 ? {num_success} : {num_errors}) * {bin_step})'
    ).mark_bar(
        opacity=0.5,
    ).encode(
        x=alt.X('bin:Q', title='Confidence', axis=alt.Axis(format='.2f')),
        y=alt.Y('density:Q', title='Relative Density', stack=None),
        color=alt.Color('label:N', legend=alt.Legend(
            title=None,
            labelExpr="datum.label === 0 ? 'Success' : 'Error'",
            legend=alt.Legend(
                title="",
                orient="none",                  
                legendX=legend_x,                   
                legendY=legend_y,                    
                direction="horizontal",
                strokeColor='silver',
                labelFontSize=12,
                padding=8
            ),
        ),
        scale=alt.Scale(range=['green', 'red']))
    ).properties(
        title=f'{metric_name}',
        width=width,
        height=height
    ).interactive()

    return chart