import pandas as pd
import numpy as np


def data_split(df,well_id_int=6110299100,param_id_list=[188,200],package_size=50000, leftcut=5000):
    
    """ Функция для разделения данных по скважинам (well_id), параметрам (param_id) и формирования пакетов
     данных,где каждый следующий пакет - конкатенация предыдущего пакета и среза заданного размера (package_size), с возможностью среза слева на определенное  число строк (leftcut) у предыдущего пакета.
     Параметры: 
     :param df: pandas DataFrame с исходными данными.
     :param well_id_int: идентификатор скважины, по которому фильтруются данные. По умолчанию 6110299100.
     :param param_id_list: список идентификаторов параметров, по которым фильтруются данные. По умолчанию [188, 200].
     :param package_size: размер пакета данных в количестве строк. По умолчанию 50000.
     :param leftcut: количество точек, которые будут обрезаны слева от каждого нового пакета. По умолчанию 5000.
     
     Возвращаемое значение: 
     :return: список пакетов данных (pandas DataFrames). """
    
    # Фильтруем данные по well_id и param_id
    df=df.query('well_id == @well_id_int and param_id in @param_id_list')
    del df['well_id'] # Удаляем столбец well_id после фильтрации

    # Устанавливаем tm_time как индекс
    df.set_index('tm_time', inplace=True)

    # Отсортируем по индексу
    df.sort_index(inplace=True)

    # Преобразование с помощью pivot, чтобы получить отдельные столбцы для каждого параметра
    df = df.pivot(columns='param_id', values='tm_value')

    # убираем верхний индекс param_id
    #df.columns.rename(None, inplace=True)

    # Интерполирование к частоте в 1 секунду
    df = df.resample('1s').mean().interpolate(method='linear', limit_direction='both')
    
    packages = [] # Создаем список для хранения пакетов данных
    current_packet = pd.DataFrame()  # Инициализируем начальный пакет как пустой DataFrame

    # Определяем общее количество точек в DataFrame    
    num_points = len(df)

    # Идем по ряду с шагом package_size
    for i in range(0, num_points, package_size):
        # Берем кусок данных длиной package_size
        package = df[i:i + package_size]

         # Добавляем новые точки к текущему пакету, оставляя только последние leftcut строк предыдущего пакета
        current_packet = pd.concat([current_packet[leftcut:], package])

        # Добавляем пакет в список
        packages.append(current_packet)

    return packages  