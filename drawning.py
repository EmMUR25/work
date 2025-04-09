import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

def draw_calculations(x, y, array_dfs, method = 'matplot'):
    """ Метод отрисовки
    
    Параметры:
    x (array): Входные данные по оси X
    y (array): Входные данные по оси Y
    array_dfs (list): лист датафреймов тренда для каждого интервала
        
    Возвращает:
    None: ничего не возвращает, показывает графики 
    """
    if method == 'plotly':
        # Создаем график
        fig = go.Figure()

        # Добавляем исходные данные
        fig.add_trace(go.Scatter(
        x=x, y=y, mode='markers', name='Исходные данные',
        marker=dict(color='blue', size=8)
        ))

        # Добавляем предсказанные значения для каждого интервала
        for i, array_df in enumerate(array_dfs):
            fig.add_trace(go.Scatter(
        x = array_df['x'], y = array_df['trend'], mode='lines',
        name=f'Тренд (интервал {i + 1})',
        line=dict(width=2)
        ))

        # Настраиваем layout
        fig.update_layout(
        title='Тренд на интервалах',
        xaxis_title='X',
        yaxis_title='Y',
        template='plotly_white'
        )

        # Показываем график
        fig.show()
    else:
        # Создаем график
        plt.figure(figsize=(10, 6))

        # Исходные данные (точечный график)
        plt.scatter(x, y, color='blue', s=30, label='Исходные данные')

        # Линии регрессии для каждого интервала
        for i, array_df in enumerate(array_dfs):
            plt.plot(
        array_df['x'], array_df['trend'], 
        linewidth=2, 
        label=f'Тренд (интервал {i + 1})'
    )

        # Настройки графика
        plt.title('Тренд на интервалах ')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()

        # Показываем график
        plt.tight_layout()
        plt.show()










def draw_PKV_segmentation(original_df, segments_df, high, low):
    """
    Оптимизированная отрисовка точек через сегменты.
    
    Параметры:
        original_df (pd.DataFrame): Исходные данные (индекс - время, колонка 'value')
        segments_df (pd.DataFrame): Сегменты с колонками ['start', 'end', 'type']
        low (float): Нижняя граница work
        high (float): Верхняя граница work
    """
    # Цвета для типов
    color_map = {
        'work': 'green',
        'down': 'red',
        'up': 'blue',
        'zero': 'black',
        'unknown': 'gray'
    }
    
    fig, ax = plt.subplots(figsize=(25, 10))
    
    # Сортируем сегменты по времени
    segments_df = segments_df.sort_values('start')
    
    # Создаем массив цветов (по умолчанию - unknown)
    colors = pd.Series('gray', index=original_df.index)
    
    # Проходим по сегментам
    for _, seg in segments_df.iterrows():
        start, end, seg_type = seg['start'], seg['end'], seg['type']
        
        # Находим все точки в этом сегменте
        mask = (original_df.index >= start) & (original_df.index <= end)
        colors.loc[mask] = seg_type
    
    # Отрисовываем все точки одним вызовом
    ax.scatter(
        x=original_df.index,
        y=original_df['value'],
        c=colors.map(color_map),
        s=50,
        alpha=0.7
    )
    
    # Подсветка диапазона work
    
    
    # Настройки графика
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Отрисовка по сегментам')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Легенда
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', 
                     markersize=15, markerfacecolor=color, label=label)
                    for label, color in color_map.items()]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()



import plotly.graph_objects as go
import pandas as pd

def draw_PKV_segmentation_plotly(original_df, segments_df, high, low):
    """
    Optimized point plotting using segments with Plotly.
    
    Parameters:
        original_df (pd.DataFrame): Original data (index - time, column 'value')
        segments_df (pd.DataFrame): Segments with columns ['start', 'end', 'type']
        low (float): Lower work boundary
        high (float): Upper work boundary
    """
    # Colors for types
    color_map = {
        'work': 'green',
        'down': 'red',
        'up': 'blue',
        'zero': 'black',
        'unknown': 'brown'
    }
    
    # Sort segments by time
    segments_df = segments_df.sort_values('start')
    
    # Create color array (default - unknown)
    colors = pd.Series('gray', index=original_df.index)
    
    # Process segments
    for _, seg in segments_df.iterrows():
        start, end, seg_type = seg['start'], seg['end'], seg['type']
        mask = (original_df.index >= start) & (original_df.index <= end)
        colors.loc[mask] = seg_type
    
    # Create figure
    fig = go.Figure()
    
    # Add all points with color mapping
    fig = go.Figure()
    fig.add_trace(
    go.Scattergl(  # вместо go.Scatter используем go.Scattergl (WebGL)
        x=original_df.index.to_numpy(),  # лучше передавать numpy array
        y=original_df['value'].to_numpy(),
        mode='markers',
        marker=dict(
            color=colors.map(color_map).to_numpy(),  # также numpy array
            size=8,
            opacity=0.7
        )
    )
)
    
   
    
    # Layout settings
    fig.update_layout(
        title='Segmentation Visualization',
        xaxis_title='Time',
        yaxis_title='Value',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600,
        width=1000,
        template="plotly_white"
    )
    
    # Custom legend for segment types
    for label, color in color_map.items():
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=10, color=color),
            name=label,
            showlegend=True
        ))
    
    fig.show()