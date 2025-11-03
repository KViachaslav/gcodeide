from svg.path import parse_path, Line, CubicBezier, Arc

def svg_path_to_points(path_string, num_samples=50):
    """
    Преобразует строку SVG-пути в массив точек (x, y).

    :param path_string: Строка из атрибута 'd' SVG-пути.
    :param num_samples: Количество точек для интерполяции на всем пути.
    :return: Массив точек в формате [(x1, y1), (x2, y2), ...].
    """
    
    # 1. Парсинг строки пути
    path = parse_path(path_string)
    
    # 2. Вычисление общей длины пути
    path_length = path.length()
    
    # 3. Генерация точек
    points = []
    
    # Определяем шаг для получения равноотстоящих точек
    step_size = path_length / (num_samples - 1) 

    for i in range(num_samples):
        # Вычисляем параметр t, где 0 <= t <= path_length
        t = i * step_size
        
        # Метод point(t) возвращает комплексное число (x + y*j),
        # где x - координата по оси X, а y - по оси Y
        complex_point = path.point(t)
        
        # Извлекаем координаты и добавляем их в массив
        x = complex_point.real
        y = complex_point.imag
        points.append((x, y))
        
    return points

# --- Пример использования ---

# Путь с абсолютными (M, L) и относительными (c, l) командами:
# M 10 10 - Move to (10, 10) (абсолютная)
# L 100 10 - Line to (100, 10) (абсолютная)
# c 0 50 50 50 50 0 - Cubic Bezier (относительная)
# l -50 0 - Line (относительная, обратно на 50 по X)
svg_d_attribute = "M 10 10 L 100 10 c 0 50 50 50 50 0 l -50 0"

array_of_points = svg_path_to_points(svg_d_attribute, num_samples=100)

print(f"Первые 5 точек: {array_of_points[:5]}")
print(f"Последние 5 точек: {array_of_points[-5:]}")

# Для работы с массивом точек можно преобразовать его, например, в NumPy
# import numpy as np
# points_array = np.array(array_of_points)