import altair as alt

from torch import Tensor
from pandas import DataFrame, melt
from typing import Union
from numpy import ndarray

def plot_train_val_loss(
    train: Union[list, ndarray, Tensor],
    val: Union[list, ndarray, Tensor],
    palette: list = ['#1b9e77', '#d95f02'], 
    title: str = 'Losses',
    width: int = 600,
    height: int = 300
) -> alt.Chart:
    
    data = DataFrame()
    data['train'] = train
    data['val'] = val
    data = data.reset_index().rename(columns={'index': 'epoch'})
    data['epoch'] = data['epoch'].apply(lambda x: x + 1)

    data = melt(
        data,
        id_vars=['epoch'],
        value_vars=['train', 'val'],
        var_name='type',
        value_name='loss'
    )

    line_chart = alt.Chart(data).mark_line().encode(
        alt.X('epoch:Q'),
        alt.Y('loss:Q'),
        alt.Color('type:N', scale=alt.Scale(range=palette)),
        tooltip=['epoch', 'type', 'loss']
    )

    point_chart = alt.Chart(data).mark_point(
        shape='circle',
        filled=True
    ).encode(
        alt.X('epoch:Q'),
        alt.Y('loss:Q'),
        alt.Color('type:N', scale=alt.Scale(range=palette)),
        tooltip=['epoch', 'type', 'loss']
    )

    chart = (line_chart + point_chart).properties(
        width=width,
        height=height,
        title=title
    )

    return chart