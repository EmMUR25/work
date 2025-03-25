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