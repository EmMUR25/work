import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline, make_splrep
from scipy import stats

def calculate_linear_reg(x, y):
    """Вычисляет линейную регрессию для заданного набора точек.
    
    Параметры:
    x (pandas.Series): Входные данные по оси X
    y (pandas.Series): Входные данные по оси Y
    
    
    Возвращает:
    DataFrame линейной регрессию для заданного набора точек
    
    """
    # Находим минимальную дату
    min_date = x.min()  
    # Преобразуем ось X из datatime в int
    x_seconds = (x - min_date).astype('timedelta64[s]').astype(int)
    
    # Строим линейную регрессию
    y_pred = stats.linregress(x_seconds, y)
    trend = y_pred.intercept + y_pred.slope * x_seconds

    # Переводим ось X обратно в dattime
    x = min_date + np.array(x_seconds, dtype='timedelta64[s]')
    
    return pd.DataFrame({'x': x, 'trend': trend})
    
def calculate_Bspline(x, y, s = 9):
    """Вычисляет B-spline для заданного набора точек.
    
    Параметры:
    x (pandas.Series): Входные данные по оси X
    y (pandas.Series): Входные данные по оси Y
    s (float): Параметр сглаживания
    
    Возвращает:
    DataFrame B-spline для заданного набора точек
    
    """
    # Строим кубический B-spline с параметром сглаживания s
    spline = make_splrep(x, y, s = s)
    new_y = spline(x)
    
    # Записываем результат в Dataframe
    spline_df = pd.DataFrame({'x': x, 'trend': new_y})
    return spline_df

def calculate_Cspline(x, y):
    """Вычисляет Cubic spline для заданного набора точек.
    
    Параметры:
    x (pandas.Series): Входные данные по оси X
    y (pandas.Series): Входные данные по оси Y
    
    
    Возвращает:
    DataFrame Cubic spline для заданного набора точек
    
    """
    # Строим CubicSpline
    spline = CubicSpline(x, y)
    new_y = spline(x)
    
    # Записываем результат в Dataframe
    spline_df = pd.DataFrame({'x': x, 'trend': new_y})
    return spline_df
    
def calculate_interval_paket(x, y, interval = 10000, method = 'linreg', s = 9):
    """Строит независимые части для каждого интервала данных.
    
    Параметры:
    x (array): Входные данные по оси X
    y (array): Входные данные по оси Y
    interval (int): Размер интервала для разбиения данных
    method (string): Название метода
    s (float): Параметр сглаживания для B-spline
    
    Возвращает:
    list: Список DataFrame с регрессиями или сплайнами для каждого интервала
    """
    results = []
       
    # Выбираем срез данных размеров interval
    for start in range(0, len(x), interval):
        end = start + interval
        # Последний срез
        if end > len(x):
            end = len(x)
          
        x_segment = x[start:end]
        y_segment = y[start:end]
        
        # Пропуск интервалов с < 2 точками
        if len(x_segment) >= 2:
            # Выбор метода
            if method == 'linreg':
                
                results.append(calculate_linear_reg(x_segment, y_segment))
            else:
                results.append(calculate_Bspline(x_segment, y_segment, s))
       
    return results if results else [pd.DataFrame({'x': x, 'trend': y})]


def calculate_from_zero_to_point_paket(x, y, interval = 18,  method = 'linreg', s = 9):
    """ Метод строится от элемента с индексом 0, до последнего через каждые spline_interval 
    
    Параметры:
    x (array): Входные данные по оси X
    y (array): Входные данные по оси Y
    interval (int): Размер интервала для разбиения данных
    method (string): Имя метода
    s (float): Параметр сглаживания для B-spline
    
    Возвращает:
    list: Список DataFrame с сплайнами или регрессиями для каждого интервала
    """
    spline_paket=[]
    
    if interval > len(x):
        return 0
        
    if interval == len(x):
        # При интервале, равном или большем длины x, применяем метод ко всему набору данных
        if method == 'linreg':
            result = calculate_linear_reg(x, y)
        else:
            result = calculate_Bspline(x, y, s)
        spline_paket.append(result)
        return spline_paket
        
    if method == 'linreg':    
        for i in range(0, len(x), interval):
            if i < 2:
                continue
            result = calculate_linear_reg(x[:i], y[:i])
            spline_paket.append(result)
       
        if len(x) != len(spline_paket[-1]):
            result = calculate_linear_reg(x, y)
            spline_paket.append(result)
        return spline_paket 
    else:
        for i in range(0, len(x), interval):
            if i < 4:
                continue
            result = calculate_Bspline(x[:i], y[:i], s)
            spline_paket.append(result)
         
        if len(x) != len(spline_paket[-1]):
            result = calculate_Bspline(x, y, s)
            spline_paket.append(result)
        return spline_paket 
