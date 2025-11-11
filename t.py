import numpy as np

def rotate_points(points, center, angle_degrees):
    """
    Поворачивает массив точек относительно заданного центра на заданный угол.

    :param points: Двумерный массив точек (N, 2), где N — количество точек.
    :param center: Кортеж или массив (2,) — координаты центра вращения (cx, cy).
    :param angle_degrees: Угол поворота в градусах (положительный — против часовой стрелки).
    :return: Новый массив повернутых точек (N, 2).
    """
    # 1. Преобразуем угол из градусов в радианы
    angle_radians = np.deg2rad(angle_degrees)
    
    # 2. Вычисляем косинус и синус угла
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    # 3. Создаем матрицу вращения 2x2
    # R = [[cos(theta), -sin(theta)], 
    #      [sin(theta),  cos(theta)]]
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # 4. Смещаем точки: P_shift = P - C
    # NumPy позволяет вычесть массив центра (1, 2) из массива точек (N, 2).
    center = np.array(center)
    points_shifted = points - center
    
    # 5. Применяем вращение: P_rotated = P_shift @ R^T (или R, в зависимости от конвенции)
    # Используем оператор @ (или np.dot) для матричного умножения.
    # points_shifted - это (N, 2), rotation_matrix - (2, 2). Результат будет (N, 2).
    points_rotated_shifted = points_shifted @ rotation_matrix.T
    
    # 6. Обратное смещение: P_final = P_rotated + C
    points_final = points_rotated_shifted + center
    
    return points_final

# --- ИСПОЛЬЗОВАНИЕ ПРИМЕРА ---

# Исходные точки (массив N x 2)
# Точка (1, 0)
# Точка (2, 1)
# Точка (1, 2)
points_array = np.array([
    [1.0, 0.0], 
    [2.0, 1.0], 
    [1.0, 2.0]
])

# Центр вращения
center_point = (1.0, 1.0) 

# Угол поворота (например, 90 градусов против часовой стрелки)
angle = 90.0

print(f"Исходные точки:\n{points_array}")
print(f"Центр вращения: {center_point}")
print(f"Угол вращения: {angle} градусов")

# Выполняем поворот
rotated_points = rotate_points(points_array, center_point, angle)

print("\nПовернутые точки (с точностью до двух знаков после запятой):")
# Ожидаемые координаты после поворота на 90 градусов вокруг (1, 1):
# (1, 0) -> (2, 1)
# (2, 1) -> (1, 2)
# (1, 2) -> (0, 1)
print(np.round(rotated_points, 2))
