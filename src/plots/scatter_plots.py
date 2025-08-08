import altair as alt

from torch import Tensor
from pandas import DataFrame
from typing import Union, Optional
from numpy import ndarray, unique, where


def plot_space(
    title: str,
    X: Union[list, ndarray, Tensor],
    Y: Union[list, ndarray, Tensor],
    Y_map: Optional[dict] = None,
    X_highlighted: Optional[Union[list, ndarray, Tensor]] = None,
    Y_highlighted: Optional[Union[list, ndarray, Tensor]] = None,
    colors: Optional[list] = ["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c", "#fdbf6f", "#ff7f00", "#cab2d6", "#6a3d9a"],
    width: int = 900,
    height: int = 500
) -> alt.Chart:

    assert X.shape[1] == 2, "X contains more of two dimensions"

    data = DataFrame(X, columns=['X1', 'X2'])
    data['Y'] = Y
    data['Index'] = data.index

    unique_labels = unique(Y)

    for label in unique_labels:
       new_label = f"Label {label}" if Y_map is None else Y_map[label]
       data.loc[data['Y'] == label, 'Y'] = new_label

    x_min, x_max = data['X1'].min(), data['X1'].max()
    y_min, y_max = data['X2'].min(), data['X2'].max()

    scatter = alt.Chart(data)

    scatter = alt.Chart(data).mark_point(
        filled=True,
        size=50,
        opacity=0.6,
        stroke='black',
        strokeWidth=0.5
    ).encode(
        alt.X('X1:Q', title='First Dimension', scale=alt.Scale(domain=(x_min, x_max))),
        alt.Y('X2:Q', title='Second Dimension', scale=alt.Scale(domain=(y_min, y_max))),
        alt.Color('Y:N', title='Label', scale=alt.Scale(range=colors)) if colors else alt.Color('Y:N', title='Label'),
        tooltip=['Index', 'X1', 'X2', 'Y']
    ).properties(
        width=width,
        height=height,
        title=title
    ).interactive()

    if (X_highlighted is not None) and (Y_highlighted is not None):

        data_highlighted = DataFrame(X_highlighted, columns=['X1', 'X2'])
        data_highlighted['Y'] = Y_highlighted

        for label in unique_labels:
            new_label = f"Label {label}" if Y_map is None else Y_map[label]
            data_highlighted.loc[data_highlighted['Y'] == label, 'Y'] = new_label
            data_highlighted['Index'] = data_highlighted.index

        scatter_highlighted = alt.Chart(data_highlighted).mark_point(
            filled=True,
            size=150,
            opacity=0.8,
            shape='diamond',
            stroke='black',
            strokeWidth=1
        ).encode(
            alt.X('X1:Q', title='First Dimension'),
            alt.Y('X2:Q', title='Second Dimension'),
            alt.Color('Y:N', title='Label', scale=alt.Scale(range=colors)) if colors else alt.Color('Y:N', title='Label'),
            tooltip=['Index', 'X1', 'X2', 'Y']
        ).properties(
            width=width,
            height=height
        )

        scatter = scatter + scatter_highlighted

    return scatter 


def plot_space_and_metric(
    metric_name: str,
    metric_values: Union[list, ndarray, Tensor],
    components: Union[list, ndarray, Tensor],
    title: str="Latent Space of Model",
    color_map: list=['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#f5f5f5'][::-1],
    width: int= 1000,
    height: int= 500
) -> alt.Chart:
    metric_name = metric_name.capitalize()

    data = DataFrame(components, columns=['X1', 'X2'])
    data[metric_name] = metric_values
    data['Index'] = data.index

    scatter = alt.Chart(data).mark_point(
        filled=True,
        stroke='black',
        strokeWidth=0.2,
        size=50,
        shape="circle"
    ).encode(
        x='X1:Q',
        y='X2:Q',
        color=alt.Color(
            f'{metric_name}:Q',
            scale=alt.Scale(range=color_map),
            legend=alt.Legend(title=f"Success - {metric_name}")
        ),
        tooltip=['X1', 'X2', metric_name, 'Index']
    ).properties(
        width=width,
        height=height,
        title=dict(
            text=title,
            subtitle=f"{metric_name} Color Map",
            fontSize=16,
            subtitleFontSize=12
        )
    ).interactive()

    return scatter


def plot_ius_and_metric(
    metric_name: str,
    metric_values: Union[list, ndarray, Tensor],
    components: Union[list, ndarray, Tensor],
    targets: Union[list, ndarray, Tensor],
    pseudo_labels: Union[list, ndarray, Tensor],
    title: str="Latent Space of Model",
    color_map_1: list=['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#f5f5f5'][::-1],
    color_map_2: list=['#f5f5f5', '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30'],
    width: int= 1000,
    height: int= 500
) -> alt.Chart:
    """
    Gera um gráfico interativo que mostra o espaço latente, destacando sucesso e erro
    de acordo com a métrica.

    Parâmetros:
        components (numpy.ndarray): Componentes (n x 2).
        pseudo_labels (numpy.ndarray): Array binário indicando erros (1) e sucessos (0).
        targets (numpy.ndarray): Rótulos originais dos dados de treinamento.
        metric_values (numpy.ndarray): Probabilidades associadas às amostras.

    Retorna:
        altair.Chart: Gráfico interativo gerado com Altair.
    """
    
    metric_name = metric_name.capitalize()

    df_data = DataFrame(components, columns=['X1', 'X2'])
    df_data['Label'] = where(pseudo_labels == 0, 'success', 'error')
    df_data['Label'] = df_data['Label'].astype('category')
    df_data['Original_label'] = targets
    df_data[metric_name] = metric_values
    df_data['Index'] = df_data.index

    success_chart = alt.Chart(df_data[df_data['Label'] == 'success']).mark_point(
        filled=True,
        stroke='black',
        strokeWidth=0.2,
        size=50,
        shape="triangle"
    ).encode(
        x='X1:Q',
        y='X2:Q',
        color=alt.Color(
            f'{metric_name}:Q',
            scale=alt.Scale(range=color_map_2),
            legend=alt.Legend(title=f"Success - {metric_name}")
        ),
        tooltip=['X1', 'X2', 'Label', metric_name, 'Original_label', 'Index']
    )

    error_chart = alt.Chart(df_data[df_data['Label'] == 'error']).mark_point(
        filled=True,
        stroke='black',
        strokeWidth=0.2,
        size=50,
    ).encode(
        x='X1:Q',
        y='X2:Q',
        color=alt.Color(
            f'{metric_name}:Q',
            scale=alt.Scale(range=color_map_1),
            legend=alt.Legend(title=f"Error - {metric_name}")
        ),
        tooltip=['X1', 'X2', 'Label', metric_name, 'Original_label', 'Index']
    )

    chart = alt.layer(success_chart, error_chart).resolve_scale(
        color='independent'
    ).properties(
        width=width,
        height=height,
        title=dict(
            text=title,
            subtitle="Success and Errors in Color Map",
            fontSize=16,
            subtitleFontSize=12
        )
    ).interactive()

    return chart
