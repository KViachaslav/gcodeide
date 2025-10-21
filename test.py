import matplotlib.pyplot as plt
from fontTools.ttLib import TTFont

from fontTools.ttLib import TTFont

# Загрузка TTF файла
font_path = 'font.ttf'
font = TTFont(font_path)

# Получение таблицы 'glyf' для доступа к контурным данным
glyf_table = font['glyf']

# Пример получения геометрии для конкретного глифа
glyph_name = 'A'  # Замените на нужное имя глифа
glyph = glyf_table[glyph_name]

# Печать контуров глифа
if glyph.isComposite():
    print(f"{glyph_name} is a composite glyph")
    for component in glyph.components:
        print(f"Component: {component}")
else:
    print(f"{glyph_name} contours: {glyph.coordinates}")
    
# x = [xx for (xx,yy) in glyph.coordinates]
# y = [yy for (xx,yy) in glyph.coordinates]


# plt.plot(x, y, color='black')
# plt.title(f"Glyph: {glyph_name}")
# plt.axis('equal')
# plt.xlim(min(x) - 10, max(x) + 10)
# plt.ylim(min(y) - 10, max(y) + 10)

# plt.show()


from shapely.geometry import Polygon, MultiPoint
from scipy.spatial import ConvexHull

# Массив точек
# points = [(0, 0), (4, 0), (4, 4), (0, 4), 
#           (1, 1), (1, 3), (3, 3), (3, 1)]
points = [(x,y) for (x,y) in glyph.coordinates]
# Преобразование в массив NumPy для использования в ConvexHull
import numpy as np
np_points = np.array(points)

# Создание выпуклой оболочки
hull = ConvexHull(np_points)

# Получение точек наружного контура
outer_contour = [tuple(np_points[i]) for i in hull.vertices]

# Получение внутренних точек
inner_points = [point for point in points if tuple(point) not in outer_contour]

# Создание полигона
polygon = Polygon(outer_contour, [inner_points])

# Вывод информации о полигоне
print("Наружный контур:", outer_contour)
print("Внутренние точки:", inner_points)
print("Полигон:", polygon)
x, y = polygon.exterior.xy

# Рисуем полигон
plt.fill(x, y, alpha=0.1, fc='blue', ec='black')
for inter in polygon.interiors:
    x, y = inter.xy

# Рисуем полигон
    plt.fill(x, y, alpha=0.1, fc='blue', ec='black')


plt.title('Polygon from Shapely')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.axis('equal')
plt.show()