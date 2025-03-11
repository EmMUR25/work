import pandas as pd
import numpy as np

def calculate_deviations(list_of_df, dict_values = {188: [50,150], 200: [20, 55]}):
    """ Функция рассчитывает различные отклонения для каждого столбца в списке DataFrame.
    Параметры: 
    :param list_of_df: Список DataFrame, содержащих временные ряды. 
    :param dict_values: Словарь, содержащий пары (ключ-значение) для каждой колонки.
    Ключ - это имя колонки (параметр), а значение - список, содержащий минимальное и максимальное установленные значения.
    Возвращаемые значения: 
    :return: Список кортежей, где каждый кортеж содержит: 
    - DataFrame с отклонениями для каждого столбца, 
    - Словарь с вычисленными статистиками для каждого столбца. """
    
    # Создаем пустой список 
    result_list=[]

    # Перебираем каждый DataFrame из списка
    for df in list_of_df:
        # Создаем пустой датафрейм для результатов
        result_df = pd.DataFrame(index=df.index)
        
        # Создаем пустой датафрейм статистик
        stats_dict = {}

        # Перебираем каждую колонку в DataFrame
        for col in df.columns:
            # Выбираем конкретный столбец
            current_column = df[col]

            # Получаем заданные минимальные и максимальные значения для текучего параметра
            set_min_value, set_max_value = dict_values.get(col)
            
            # Вычисление среднего, медианы, квартилей и стандартное отклонение для заданного столбца
            mean_value = current_column.mean()
            median_value = current_column.median()
            quartiles = current_column.quantile([0.25, 0.75])
            std_dev = current_column.std()
            
            # Параметры арифметической прогрессии для весов
            a1 = 1     # Первый член прогрессии
            d = 2      # Разность прогрессии
            # Генерация весов как арифметической прогрессии
            weights = [a1 + (i)*d for i in range(len(df.index))]

            # Расчет средневзвешенного значения
            weighted_mean = sum(current_column * weights) / sum(weights)

            
            # Добавляем статистики в словарь
            stats_dict[col] = {
            'mean': mean_value,
            'weighted_mean': weighted_mean,
            'Q1': quartiles.loc[0.25],
            'median': median_value,
            'Q3': quartiles.loc[0.75],
            'std': std_dev,
            'set_min': set_min_value,
            'set_max': set_max_value
                }
            # Столбец данных
            result_df[f'{col}_Value'] = current_column
            
            # Отклонения от среднего
            result_df[f'{col}_Deviation from Mean'] = current_column - mean_value
            
            # Отклонения от средневзвешенного
            result_df[f'{col}_Deviation from Weighted Mean'] = current_column - weighted_mean
        
            # Отклонения от первого квартиля
            result_df[f'{col}_Deviation from Q1'] = current_column - quartiles.loc[0.25]
        
            # Отклонения от медианы
            result_df[f'{col}_Deviation from Median'] = current_column - median_value
        
            # Отклонения от третьего квартиля
            result_df[f'{col}_Deviation from Q3'] = current_column - quartiles.loc[0.75]
        
            # Отклонения от предыдущего значения
            result_df[f'{col}_Deviation from Previous Value'] = current_column.diff()

            # Отклонения от min значения
            result_df[f'{col}_Deviation from Set Min Value'] = current_column - set_min_value

            # Отклонения от max значения
            result_df[f'{col}_Deviation from Set Max Value'] = current_column - set_max_value
            
    
        result_list.append([result_df,stats_dict])
        
    return result_list