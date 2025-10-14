# import shapely
# from shapely import MultiPoint, Polygon

# import matplotlib.pyplot as plt
# from shapely.geometry import Polygon
# multi_point = MultiPoint([(0, 0), (0, 3), (1, 1), (3, 0), (3, 3)])

# polygon = shapely.concave_hull(multi_point, ratio=0)

# x, y = polygon.exterior.xy

# plt.figure(figsize=(6, 6))
# plt.fill(x, y, alpha=0.5, fc='lightblue', ec='black')
# plt.title('Polygon Visualization')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.grid()
# plt.axhline(0, color='black', linewidth=0.5, ls='--')
# plt.axvline(0, color='black', linewidth=0.5, ls='--')
# plt.xlim(0, 6)
# plt.ylim(0, 5)
# plt.show()


import numpy as np

def distance_point_to_segment(px, py, ax, ay, bx, by):
    # Превращаем точки в numpy массивы
    p = np.array([px, py])
    a = np.array([ax, ay])
    b = np.array([bx, by])
    
    # Вектор AB
    ab = b - a
    # Вектор AP
    ap = p - a
    # Вектор BP
    bp = p - b

    # Длина AB в квадрате
    ab_len_sq = np.dot(ab, ab)
    
    # Если A и B совпадают
    if ab_len_sq == 0:
        return np.linalg.norm(ap)  # расстояние от точки до A (или B)

    # Параметр t для проекции P на AB
    t = np.dot(ap, ab) / ab_len_sq
    t = np.clip(t, 0, 1)  # ограничиваем t от 0 до 1
    
    # Находим ближайшую точку на отрезке
    nearest = a + t * ab
    
    # Возвращаем расстояние от P до ближайшей точки
    return np.linalg.norm(p - nearest)

# Пример использования
px, py = 8, 3  # координаты точки
ax, ay = 1, 1  # координаты начала отрезка
bx, by = 4, 4  # координаты конца отрезка

dist = distance_point_to_segment(px, py, ax, ay, bx, by)
print(f"Расстояние от точки до отрезка: {dist}")