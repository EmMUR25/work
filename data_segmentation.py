import pandas as pd
from statistics import median

import math

def count_sequences(values):
    """Подсчитывает количество возрастающих (up) и убывающих (down) последовательностей в списке значений.
    
    Args:
        values: Список числовых значений для анализа.
        
    Returns:
        Кортеж из двух элементов:
        - Словарь с количеством последовательностей ('up' и 'down').
        - Список кортежей с информацией о каждой последовательности (начальный индекс, конечный индекс, тип).
    """

     # Обработка пустого списка
    if not values:
        return {'up': 0, 'down': 0}
    
    n = len(values)
    # Список из одного элемента не содержит последовательностей
    if n <= 1:
        return {'up': 0, 'down': 0}
    
    sequences = [] # Хранит информацию о найденных последовательностях
    start = 0 # Индекс начала текущей последовательности
    current_type = None # Тип текущей последовательности ('up' или 'down')
    prev = values[0] # Предыдущее значение для сравнения
    counts = {'up': 0, 'down': 0} # Счетчики последовательностей
    
    for i in range(1, n):
        current = values[i]
        # Определяем направление изменения между текущим и предыдущим значением
        if current > prev:
            direction = 'up'
        elif current < prev:
            direction = 'down'
        else:
            direction = 'flat' # Значения равны - направление не меняется
        
        # Если тип последовательности еще не определен (начало)
        if current_type is None:
            if direction == 'up':
                current_type = 'up'
            elif direction == 'down':
                current_type = 'down'
        else:
            # Если было возрастание, а теперь убывание - фиксируем последовательность
            if current_type == 'up' and direction == 'down':
                counts[current_type] += 1
                sequences.append((start, i-1, current_type))
                start = i # Новая последовательность начинается с текущего индекса
                current_type = 'down'
            # Если было убывание, а теперь возрастание - фиксируем последовательность    
            elif current_type == 'down' and direction == 'up':
                counts[current_type] += 1
                sequences.append((start, i-1, current_type))
                start = i
                current_type = 'up'
        
        prev = current # Обновляем предыдущее значение

    # Фиксируем последнюю последовательность
    if current_type is not None:
        counts[current_type] += 1
        sequences.append((start, n-1, current_type))
    else:
        counts['up'] += 1
        sequences.append((0, n-1, 'up'))
    
    
    return counts, sequences

def flat_seq_work_places(values):
    """Находит начало и конец рабочего периода, где значения одинаковы , не нулевые и больше 25.
    
    Рабочий период определяется как последовательность одинаковых значений (>25 и !=0)
        
    Args:
        values: Список числовых значений, представляющих показатели рабочих мест.
        
    Returns:
        Кортеж (start_work, end_work) - индексы начала и конца рабочего периода,
        или None, если такой период не найден.
    """
    start_work=None
    end_work=None

    # Ищем начало рабочего периода (первое вхождение двух одинаковых значений >25)
    for i in range(1,len(values)):
        if values[i] == values[i-1] and values[i] !=0 and values[i] > 25:
            start_work = i 
            
            break
    # Ищем конец рабочего периода (последнее вхождение двух одинаковых значений >25)    
    for i in range(len(values) - 2, -1, -1):
        if values[i] == values[i+1] and values[i] !=0 and values[i] >25:
            end_work = i
     # Если не нашли начало или конец - возвращаем None        
    if start_work == None or end_work == None:
        return None
    else:
        return start_work, end_work
    
def find_work_places(values):
    """Находит рабочие области на основе анализа последовательностей значений.
    
    Использует две стратегии:
    1. При сложном изменении значений - возрастаний>1 и убываний>1 анализирует последовательности
    2. В другом случае использует функцию flat_seq_work_places() 
    
    Args:
        values: Список числовых значений
        
    Returns:
        Кортеж (start, end) с индексами рабочего периода или None, если период не найден
    """
    # Получаем информацию о последовательностях из count_sequence
    counts, seq = count_sequences(values)
    
    # Инициализируем переменные для границ рабочего периода
    work_places_start = 0
    work_places_end = 0
    
    
    if counts['up'] > 1 and counts['down'] > 1:
        # Находим конец первой убывающей последовательности
        first_end_down = 0
        for i in seq:
            if i[2] == 'down':
                first_end_down = i[1]
                
                break
        # Обработка случаев, когда количество последовательностей >=2 каждого типа
        # и значение в конце первой убывающей последовательности <= 25
        if counts['up'] >= 2 and counts['down'] >= 2 and values[first_end_down] <= 25:
            # Находим конец второй возрастающей последовательности
            second_end_up = 0
            for i in seq:
                if i[2] == 'up':
                    second_end_up += 1
                    if second_end_up == 2:
                        second_end_up = i[1]
                        break
            work_places_start = second_end_up
        else:
            # Иначе берем конец первой возрастающей последовательности
            first_end_up = 0
            for i in seq:
                if i[2] == 'up':
                    first_end_up = i[1]
                    break
            work_places_start = first_end_up
        first_end_down = 0
        for i in range(len(seq) - 1, -1, -1):
            if seq[i][2] == 'down':
                first_end_down = seq[i][0]
                break

        # Временные переменные    
        second_end_up = 0
        first_end_up = 0

        # Аналогичная проверка для конца рабочего периода
        if counts['up'] >= 2 and counts['down'] >= 2 and values[first_end_down] <= 25: 
            # Ищем конец предпоследней возрастающей последовательности
            for i in range(len(seq) - 1, -1, -1):
                if seq[i][2] == 'up':
                    second_end_up += 1
                    if second_end_up == 2:
                        second_end_up = seq[i][1]
                        break
                        
            work_places_end = second_end_up
        # Иначе берем конец последней возрастающей последовательности    
        else:
            for i in range(len(seq) - 1, -1, -1):
                if seq[i][2] == 'up':
                    first_end_up = seq[i][1]
                    break
            work_places_end = first_end_up
        # Проверка найденного периода    
        if work_places_start == work_places_end:
            return None
        else:
            return work_places_start, work_places_end
    else:
        # Для случаев когда число up и (или) down сегментов = 1 используем другой алгоритм
        return flat_seq_work_places(values)
    
def find_work_stats_df(df, column_name):
    """
    Обрабатывает DataFrame с индексом datetime и возвращает статистики рабочей части для указанной колонки.
    
    Параметры:
    - df: pandas.DataFrame с индексом datetime.
    - column_name: str, название колонки для анализа.
    
    Возвращает:
    - dict или None: словарь с ключами 'min', 'max', 'avg', либо None, если данные некорректны.
    """
    if column_name not in df.columns:
        raise ValueError(f"Колонка '{column_name}' не найдена в DataFrame")
    
    values = df[column_name].tolist()
    values = [round_down(x,2.0) for x in values]
    
    
    result = find_work_places(values)
    
    if result is None:
        return None
      
    res0= result[0]
    res1= result[1]
    
    if res0 > res1:
        res0,res1=res1,res0

    
    val = values[res0:res1+1]
    
    if result is None:
        return None
    else:
        return {
            'min': min(val),
            'max': max(val),
            'avg': median(val)
        }
    
def find_zero_to_zero_df(df, column_name):
    """Находит индексы первого и второго нуля, между которыми есть возрастание и убывание.
    
    Параметры:
    - df: pandas.DataFrame с индексом datetime.
    - column_name: str, название колонки для анализа.
    
    Возвращает:
    - Кортеж (первый_ноль, второй_ноль) в формате datetime или (None, None), если не найдено.
    """
    if column_name not in df.columns:
        raise ValueError(f"Колонка '{column_name}' не найдена в DataFrame")
    
      
    values = df[column_name].tolist()
    
    first_zero_idx = None
    second_zero_idx = None
    
    # Поиск первого нуля, после которого значение возрастает
    for i in range(len(values)):
        if values[i] == 0:
            first_zero_idx = i 
            break  # Найден первый ноль
    
    # Если первый ноль не найден, возвращаем None
    if first_zero_idx is None:
        return None,None
    
    # Поиск второго нуля после первого, где значение убывает до нуля
    for i in range(first_zero_idx + 1, len(values)):
        if values[i] < values[i-1] and values[i] == 0:
            second_zero_idx = i
            break  # Найден второй ноль
            
    if second_zero_idx is None:
        return None, None
    # Получаем метки времени из индекса
    first_zero_time = df.index[first_zero_idx] 
    second_zero_time = df.index[second_zero_idx] 
    
    return (first_zero_time, second_zero_time)    


def round_down(n, decimals=0):
    """Округляет число n вниз до указанного количества десятичных знаков.

    Параметры:
        n (float): Число, которое нужно округлить.
        decimals (int, optional): Количество десятичных знаков после запятой. 
                                  По умолчанию 0 (округляет до целого числа).

    Возвращает:
        float: Число n, округленное вниз до decimals десятичных знаков.
    """
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

def short_work_segments(segments_df, min_work_duration):
    """Обрабатывает короткие рабочие сегменты в DataFrame, объединяя их с соседними сегментами другого типа.

    Параметры:
        segments_df (pd.DataFrame): DataFrame с сегментами, должен содержать колонки:
                                    - 'type': тип сегмента ('work', 'zero' или другие)
                                    - 'duration': длительность сегмента
                                    - 'start': начало сегмента
                                    - 'end': конец сегмента
        min_work_duration (float): Минимальная допустимая длительность для рабочих ('work') сегментов.

    Возвращает:
        pd.DataFrame: Обработанный DataFrame с объединенными сегментами.
    """

    # Если DataFrame пуст, возвращаем его как есть
    if len(segments_df) == 0:
        return segments_df
    
    adjusted_segments = []  # Список для хранения обработанных сегментов
    i = 0
    n = len(segments_df)
    
    while i < n:
        current_seg = segments_df.iloc[i].copy() # Текущий сегмент
        
         # Проверяем, является ли текущий сегмент рабочим и слишком коротким
        if current_seg['type'] == 'work' and current_seg['duration'] < min_work_duration:
            
            # Ищем предыдущий не-'zero' сегмент (если есть)
            prev_type = None
            j = i - 1
            while j >= 0 and prev_type is None:
                if segments_df.iloc[j]['type'] != 'zero':
                    prev_type = segments_df.iloc[j]['type']
                j -= 1
            
            # Если не нашли предыдущий не-'zero' тип, ищем следующий
            if prev_type is None:
                j = i + 1
                while j < n and prev_type is None:
                    if segments_df.iloc[j]['type'] != 'zero':
                        prev_type = segments_df.iloc[j]['type']
                    j += 1
            
                      
             # Меняем тип текущего сегмента на найденный (prev_type)
            current_seg['type'] = prev_type
            
            # Пытаемся объединить с соседними сегментами того же типа
            # Проверяем предыдущий сегмент в adjusted_segments
            if adjusted_segments and adjusted_segments[-1]['type'] == prev_type:
                # Объединяем с предыдущим
                merged_seg = adjusted_segments[-1].copy()
                merged_seg['end'] = current_seg['end']
                merged_seg['duration'] = merged_seg['end'] - merged_seg['start']
                adjusted_segments[-1] = merged_seg
                i += 1
                continue
            
            # Проверяем следующий сегмент в исходном DataFrame
            if i + 1 < n and segments_df.iloc[i+1]['type'] == prev_type:
                # Объединяем со следующим
                current_seg['end'] = segments_df.iloc[i+1]['end']
                current_seg['duration'] = current_seg['end'] - current_seg['start']
                adjusted_segments.append(current_seg)
                i += 2
                continue
        # Если не было объединения, просто добавляем текущий сегмент
        adjusted_segments.append(current_seg)
        i += 1
    # Возвращаем результат в виде DataFrame
    return pd.DataFrame(adjusted_segments)

def determine_type(value, prev_value, max_work_value, min_work_value):
    """Определяет тип значения на основе текущего значения, предыдущего значения 
    и границ рабочего диапазона.

    Параметры:
        value (float/int): Текущее значение для анализа
        prev_value (float/int): Предыдущее значение (может быть None)
        max_work_value (float/int): Максимальное значение рабочего диапазона (может быть None)
        min_work_value (float/int): Минимальное значение рабочего диапазона (может быть None)

    Возвращает:
        str: Тип значения:
            - 'zero' - нулевое значение
            - 'work' - значение в рабочем диапазоне
            - 'up' - значение растет по сравнению с предыдущим
            - 'down' - значение уменьшается по сравнению с предыдущим
            - 'work_like' - значение похоже на рабочее (>25), но не в рабочем диапазоне
            - 'unknown' - тип не удалось определить
    """

    # Если значение равно 0 - возвращаем 'zero'
    if value == 0:
        return 'zero'
    # Если заданы оба граничных значения рабочего диапазона и они больше 25
    elif min_work_value!= None and max_work_value!= None and max_work_value > 25 and min_work_value >25:
        # Специальный случай, когда min и max равны (фиксированное значение)
        if min_work_value == max_work_value and min_work_value <= value:
            return "work"
        # Если значение попадает в рабочий диапазон
        if min_work_value <= value <= max_work_value:
            return 'work'
        else:
            # Если предыдущего значения нет - возвращаем 'unknown'
            if prev_value is None:
                return 'unknown'
            # Если значение растет
            elif value>prev_value:
                return 'up' 
            # Если значение уменьшается
            elif prev_value > value:
                return 'down'
            # Если значение больше 25, но не в рабочем диапазоне
            elif value >25:
                return 'work_like'
            # Во всех остальных случаях
            else:
                return 'unknown'
    # Если границы рабочего диапазона не заданы или некорректны        
    else:
        # Если предыдущего значения нет - возвращаем 'unknown'
        if prev_value is None:
            return 'unknown'
        # Если значение растет
        elif value > prev_value:
            return 'up'
        # Если значение уменьшается
        elif prev_value > value:
            return 'down'
        # Если значение больше 25, но нет данных о рабочем диапазоне
        elif value > 25:
            return 'work_like'
        # Во всех остальных случаях
        else:
            return 'unknown'
        


def PKV_segmentation(df, max_work_value, min_work_value):
    """Сегментирует временной ряд на участки разных типов на основе значений и рабочих диапазонов.
    
    Параметры:
        df (pd.DataFrame): Входной DataFrame с временным рядом
        max_work_value (float): Максимальное значение рабочего диапазона
        min_work_value (float): Минимальное значение рабочего диапазона
    
    Возвращает:
        pd.DataFrame: DataFrame с сегментами, содержащий колонки:
                     - start: начало сегмента
                     - end: конец сегмента
                     - duration: длительность сегмента
                     - type: тип сегмента
    """
        
    results = [] # Список для хранения результирующих сегментов
    deferred_point = None  # (value, time, type)
    prev_value = None # Предыдущее значение
        
    # Основной цикл по точкам временного ряда    
    for idx, point_value in df.iloc[:, 0].items():
        # Обработка отложенной точки (если есть)
        if deferred_point:
            d_value, d_time, d_type = deferred_point
            # Определяем тип для отложенной точки
            resolved_type = 'up' if point_value > d_value else 'down'
            # Определяем тип для текущей точки
            point_type = determine_type(point_value, d_value, max_work_value, min_work_value)
            # Если тип совпадает с текущим, расширяем сегмент
            if resolved_type == point_type:
                results.append({
                    'start': d_time,
                    'end': idx,
                    'duration': idx - d_time,
                    'type': point_type
                })
            else: # Иначе создаем два отдельных сегмента
                results.append({
                    'start': d_time,
                    'end': d_time,
                    'duration': pd.Timedelta(0),
                    'type': resolved_type
                })
                results.append({
                    'start': idx,
                    'end': idx,
                    'duration': pd.Timedelta(0),
                    'type': point_type
                })
            deferred_point = None 
            prev_value = point_value
            
            continue
        # Определение типа точки    
        point_type = determine_type(point_value, prev_value, max_work_value, min_work_value)
        # Обработка типа точки unknow
        if point_type == 'unknown':
            deferred_point = (point_value, idx, point_type)
        else: # Если тип есть
            if not results: # Если первая точка
                results.append({
                    'start': idx,
                    'end': idx,
                    'duration': pd.Timedelta(0),
                    'type': point_type
                })
            else:  
                last_segment = results[-1] # Тот же тип
                if last_segment['type'] == point_type:
                    last_segment['end'] = idx
                    last_segment['duration'] = idx - last_segment['start']
                else: 
                    results.append({
                        'start': idx,
                        'end': idx,
                        'duration': pd.Timedelta(0),
                        'type': point_type
                    })
        prev_value = point_value
        
    
    if deferred_point:
        _, d_time, _ = deferred_point
        results.append({
            'start': d_time,
            'end': d_time,
            'duration': pd.Timedelta(0),
            'type': 'unknown'
        })
    # Объединение сегментов между первым и последним work, если между ними нет zero
    work_indices = [i for i, seg in enumerate(results) if seg['type'] == 'work']
    
    if len(work_indices) >= 2:
        first_work = work_indices[0]
        last_work = work_indices[-1]
        
        # Проверяем, есть ли zero между first_work и last_work
        has_zero_between = any(
            seg['type'] == 'zero'
            for seg in results[first_work + 1 : last_work]
        )
        
        # Если zero нет — объединяем
        if not has_zero_between:
            merged_segment = {
                'start': results[first_work]['start'],
                'end': results[last_work]['end'],
                'duration': results[last_work]['end'] - results[first_work]['start'],
                'type': 'work'  # Можно заменить на 'work', если нужно
            }
            
            # Заменяем диапазон на один объединенный сегмент
            results = results[:first_work] + [merged_segment] + results[last_work + 1:]
    return pd.DataFrame(results)

def determine_type_online(point_value, prev_value, was_work):
    """Определяет тип точки в онлайн-режиме на основе текущего и предыдущего значений.
    
    Параметры:
        point_value (float): Текущее значение
        prev_value (float): Предыдущее значение
        was_work (bool): Флаг, был ли предыдущий сегмент рабочим
        
    Возвращает:
        tuple: (тип точки, флаг was_work)
    """
    # Вычисление производной
    derivative = (point_value - prev_value)

    sign = 1 if derivative > 0 else (-1 if derivative < 0 else 0)

    angle = math.degrees(math.atan(derivative))  
    # Определение типа точки
    if point_value == 0:
        return 'zero', False
    
    elif sign == 1:
        if (angle > 1 and was_work == False) or (point_value < 25):  # деленеие и углы + перейти к процентам 
            return 'up', was_work  # вынесение консант (константы алгоритмов и id параматров)
        else:
            return 'work', True
    elif sign == -1:
        if abs(angle) > 1 or point_value < 25:
            return 'down', was_work
        else:
            return 'work', True
    else:
        return 'work', True

def PKV_segmentation_online(df):
    """Онлайн-сегментация временного ряда на участки разных типов.
    
    Параметры:
        df (pd.DataFrame): Входной DataFrame с временным рядом
        
    Возвращает:
        pd.DataFrame: DataFrame с сегментами, содержащий колонки:
                     - start: начало сегмента
                     - end: конец сегмента
                     - duration: длительность сегмента
                     - type: тип сегмента
    """
    results = [] # Список для хранения сегментов
    deferred_point = None # Отложенная точка для обработки
    prev_value = None # Предыдущее значение
    was_work = False # Флаг рабочего состояния
    max_down_point = 0 # Максимальное значение при падении
    max_up_point = 0 # Максимальное значение при росте
    
    # Первый проход - обычная сегментация
    for idx, point_value in df.iloc[:, 0].items():
        if deferred_point:
            d_value, d_time = deferred_point
            # Определяем тип для отложенной точки
            resolved_type = 'up' if point_value > d_value else ('down' if point_value < d_value else 'zero')
            
            # Определяем тип для текущей точки
            point_type, was_work = determine_type_online(point_value, prev_value, was_work)
            
            # Если типы совпадают - создаем объединенный сегмент
            if resolved_type == point_type:
                segment = {'start': d_time, 'end': idx, 'duration': idx - d_time, 'type': point_type}
                results.append(segment)
            else:
                # Иначе создаем два отдельных сегмента
                results.append({'start': d_time, 'end': d_time, 'duration': pd.Timedelta(0), 'type': resolved_type})
                results.append({'start': idx, 'end': idx, 'duration': pd.Timedelta(0), 'type': point_type})
                
            deferred_point = None
            prev_value = point_value
            continue
        # Определение типа точки
        if prev_value is not None:
            point_type, was_work = determine_type_online(point_value, prev_value, was_work) 
        else:
            point_type = 'unknown'

        # Обработка точки с неизвестным типом
        if point_type == 'unknown':
            deferred_point = (point_value, idx)
        else:
            if not results: # Если это первая точка
                results.append({'start': idx, 'end': idx, 'duration': pd.Timedelta(0), 'type': point_type})
            else:
                last_segment = results[-1]
                # Если тип совпадает с предыдущим сегментом - расширяем его
                if last_segment['type'] == point_type:
                    last_segment['end'] = idx
                    last_segment['duration'] = idx - last_segment['start']
                else:
                    # Иначе создаем новый сегмент
                    results.append({'start': idx, 'end': idx, 'duration': pd.Timedelta(0), 'type': point_type})

        prev_value = point_value
       
    
        # Второй проход - преобразование сегментов
        if len(results) < 3: 
            continue
        i = 0
        while i < len(results):
            
            if i > 0 and i < len(results)-1:
            # Правило 1: если work < 5 секунд  а до и после него одинаковый тип то work -> up или down
                if (results[i-1]['type'] == results[i+1]['type'] and results[i]['type'] == 'work' and results[i]['duration'] < pd.Timedelta(seconds= 5)):
                    results[i-1]['end'] = results[i+1]['end']
                    results[i-1]['duration'] = results[i-1]['end'] - results[i-1]['start']
                    
                    del results[i:i+2]
                    was_work = False
                    i -= 1

            if i > 0 and i < len(results)-1:
            # Правило 2: down между up и work -> work    
                if (results[i-1]['type'] == 'up' and results[i]['type'] == 'down' and results[i+1]['type'] == 'work'):
                    results[i]['type'] = 'work'
        
            # Правило 3: любые сегменты между work и work -> work с объединением
                if (results[i-1]['type'] == 'work' and results[i+1]['type'] == 'work'):
                # Преобразуем текущий сегмент в work
                    results[i]['type'] = 'work'
                
                # Объединяем все три сегмента в один
                    results[i-1]['end'] = results[i+1]['end']
                    results[i-1]['duration'] = results[i-1]['end'] - results[i-1]['start']
                
                # Удаляем лишние сегменты
                    del results[i:i+2]
                    i -= 1  # Возвращаемся назад, так как изменили структуру списка
                
        
            i += 1
        

        # Третий проход - окончательное объединение соседних work сегментов
        i = 1
        while i < len(results):
            if results[i-1]['type'] == 'work' and results[i]['type'] == 'work':
                results[i-1]['end'] = results[i]['end']
                results[i-1]['duration'] = results[i-1]['end'] - results[i-1]['start']
                del results[i]
            else:
                i += 1
        # Четвертый проход поиск down - up
        i = 1
        while i < len(results):
            if results[i-1]['type'] == 'down' and results[i]['type'] == 'up' :
                max_down_point = df.loc[results[i-1]['start']].value
                max_up_point = df.loc[results[i]['end']].value
                if max_down_point >= max_up_point:
                    results[i-1]['end'] = results[i]['end']
                    results[i-1]['duration'] = results[i-1]['end'] - results[i-1]['start']
                    results[i-1]['type'] = 'work'
                    was_work = True
                    del results[i]
                    
            else:
                i+=1        
    return pd.DataFrame(results)

import matplotlib.pyplot as plt
import math
from PIL import Image
import io
import os
from matplotlib.colors import to_rgba
import imageio 

def determine_type_online_gif(point_value, prev_value, was_work):
    """Определяет тип точки для визуализации в GIF анимации.
    
    Параметры:
        point_value (float): текущее значение точки
        prev_value (float): предыдущее значение
        was_work (bool): был ли предыдущий сегмент рабочим
        
    Возвращает:
        tuple: (тип точки, обновленный флаг was_work)
    """
    # Вычисляем производную
    derivative = (point_value - prev_value)

    # Вычисляем угол наклона в градусах
    sign = 1 if derivative > 0 else (-1 if derivative < 0 else 0)
    angle = math.degrees(math.atan(derivative))
    
    # Логика определения типа точки
    if point_value == 0:
        return 'zero', False
    elif sign == 1:  # Растущие значения
        if (angle > 1 and was_work == False) or (point_value < 25):
            return 'up', was_work
        else:
            return 'work', True
    elif sign == -1: # Падающие значения
        if abs(angle) > 1 or point_value < 25:
            return 'down', was_work
        else:
            return 'work', True
    else: # Нулевая производная
        return 'work', True

def PKV_segmentation_online_gif(df, output_dir="segmentation_frames"):
    """Выполняет онлайн-сегментацию с сохранением кадров для GIF анимации.
    
    Параметры:
        df (pd.DataFrame): входные данные временного ряда
        output_dir (str): директория для сохранения кадров анимации
        
    Возвращает:
        pd.DataFrame: результат сегментации
    """
    
    # Создаем папку для кадров, если ее нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Очищаем папку, если она уже существует
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))

    # Инициализация переменных
    results = []
    deferred_point = None
    prev_value = None
    was_work = False
    max_down_point = 0
    max_up_point = 0
    frame_count = 0
    
    # Основной цикл обработки точек
    for idx, point_value in df.iloc[:, 0].items():
        # Обработка отложенной точки
        if deferred_point:
            d_value, d_time = deferred_point
            resolved_type = 'up' if point_value > d_value else ('down' if point_value < d_value else 'zero')
            
            point_type, was_work = determine_type_online_gif(point_value, prev_value, was_work)
            
            if resolved_type == point_type:
                segment = {'start': d_time, 'end': idx, 'duration': idx - d_time, 'type': point_type}
                results.append(segment)
            else:
                results.append({'start': d_time, 'end': d_time, 'duration': pd.Timedelta(0), 'type': resolved_type})
                results.append({'start': idx, 'end': idx, 'duration': pd.Timedelta(0), 'type': point_type})
                
            deferred_point = None
            prev_value = point_value
            continue
        
        # Определение типа текущей точки
        if prev_value is not None:
            point_type, was_work = determine_type_online_gif(point_value, prev_value, was_work) 
        else:
            point_type = 'unknown'
        # Обработка неизвестного типа
        if point_type == 'unknown':
            deferred_point = (point_value, idx)
        else:
            if not results:
                results.append({'start': idx, 'end': idx, 'duration': pd.Timedelta(0), 'type': point_type})
            else:
                last_segment = results[-1]
                if last_segment['type'] == point_type:
                    last_segment['end'] = idx
                    last_segment['duration'] = idx - last_segment['start']
                else:
                    results.append({'start': idx, 'end': idx, 'duration': pd.Timedelta(0), 'type': point_type})

        prev_value = point_value
        
        # Визуализация текущего состояния
        if len(results) > 0:
         # Фиксируем размер и масштаб графика
            fig, ax = plt.subplots(figsize=(12, 6))
    
             # Устанавливаем фиксированные границы по осям (если данные уже есть)
            if not df.empty:
                ax.set_xlim(df.index.min(), df.index.max())
                y_min = df.iloc[:, 0].min() - (df.iloc[:, 0].max() - df.iloc[:, 0].min()) * 0.1
                y_max = df.iloc[:, 0].max() + (df.iloc[:, 0].max() - df.iloc[:, 0].min()) * 0.1
                ax.set_ylim(y_min, y_max)
    
        # Создаем легенду
            legend_elements = [
                
            plt.Line2D([0], [0], marker='o', color='w', label='Up', 
                   markerfacecolor='blue', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='Down', 
                   markerfacecolor='red', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='Zero', 
                   markerfacecolor='black', markersize=8),
            plt.Line2D([0], [0], marker='o', color='w', label='Work', 
                   markerfacecolor='green', markersize=8)
                      ]
            ax.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 1.15), ncol=4)
    
            # Визуализация сегментов
            for segment in results:
                segment_data = df.loc[segment['start']:segment['end']]
                color = {
            'up': 'blue',
            'down': 'red',
            'zero': 'black',
            'work': 'green'
        }.get(segment['type'], 'gray')
        
        # Только точки (без линий)
                ax.plot(segment_data.index, segment_data.iloc[:, 0], 'o', 
                        color=color, markersize=5, alpha=0.7)
        
        # Подпись типа сверху
                ax.text(segment['start'], y_max * 1.02, segment['type'],
                ha='left', va='bottom', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Настройки отображения
            ax.set_title(f"Segmentation State (Point {idx})", pad=20)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            plt.tight_layout()
    
    # Сохранение с фиксированными параметрами
            frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0.2)
            plt.close(fig)
            frame_count += 1
        
        if len(results) < 3: 
            
            continue
        i = 0
       
        while i < len(results):
            
            if i > 0 and i < len(results)-1:
            # Правило 1: если work < 5 секунд  а до и после него одинаковый тип то work -> up или down
                if (results[i-1]['type'] == results[i+1]['type'] and results[i]['type'] == 'work' and results[i]['duration'] < pd.Timedelta(seconds= 5)):
                    results[i-1]['end'] = results[i+1]['end']
                    results[i-1]['duration'] = results[i-1]['end'] - results[i-1]['start']
                    
                    del results[i:i+2]
                    was_work = False
                    i -= 1

            if i > 0 and i < len(results)-1:
            # Правило 2: down между up и work -> work    
                if (results[i-1]['type'] == 'up' and results[i]['type'] == 'down' and results[i+1]['type'] == 'work'):
                    results[i]['type'] = 'work'
        
            # Правило 3: любые сегменты между work и work -> work с объединением
                if (results[i-1]['type'] == 'work' and results[i+1]['type'] == 'work'):
                # Преобразуем текущий сегмент в work
                    results[i]['type'] = 'work'
                
                # Объединяем все три сегмента в один
                    results[i-1]['end'] = results[i+1]['end']
                    results[i-1]['duration'] = results[i-1]['end'] - results[i-1]['start']
                
                # Удаляем лишние сегменты
                    del results[i:i+2]
                    i -= 1  # Возвращаемся назад, так как изменили структуру списка
                
            
            i += 1
        
        # Третий проход - окончательное объединение соседних work сегментов
        i = 1
        while i < len(results):
            if results[i-1]['type'] == 'work' and results[i]['type'] == 'work':
                results[i-1]['end'] = results[i]['end']
                results[i-1]['duration'] = results[i-1]['end'] - results[i-1]['start']
                del results[i]
            else:
                i += 1
        
        # Четвертый проход поиск down - up
        i = 1
        while i < len(results):
            
            if results[i-1]['type'] == 'down' and results[i]['type'] == 'up' :
                max_down_point = df.loc[results[i-1]['start']].value
                max_up_point = df.loc[results[i]['end']].value
                if max_down_point >= max_up_point:
                    results[i-1]['end'] = results[i]['end']
                    results[i-1]['duration'] = results[i-1]['end'] - results[i-1]['start']
                    results[i-1]['type'] = 'work'
                    was_work = True
                    del results[i]
            else:
                i+=1
       
        result_prev=results
           
    return pd.DataFrame(results)

def create_segmentation_gif(frame_dir="segmentation_frames", output_path="segmentation_process.gif"):
    """Создает GIF из сохраненных кадров"""
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.startswith('frame_') and f.endswith('.png')])
    
    if not frame_files:
        print("Нет кадров для создания GIF")
        return
    
    with imageio.get_writer(output_path, mode='I', duration=0.3) as writer:
        for frame_file in frame_files:
            frame_path = os.path.join(frame_dir, frame_file)
            img = imageio.imread(frame_path)
            writer.append_data(img)
    
    print(f"GIF успешно создан: {output_path}")