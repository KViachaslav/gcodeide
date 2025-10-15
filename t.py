import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# Определяем два полигона
points1 = [(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)]
points2 = [(4, 4), (4, 0), (3, 0), (3, 3), (0, 3), (0, 4)]

# Создаем полигоны
polygon1 = Polygon(points1)
polygon2 = Polygon(points2)

# Объединяем полигоны
merged_polygon = polygon1.union(polygon2)

# Создаем фигуру и оси
fig, ax = plt.subplots()

# # Рисуем первый полигон
# x1, y1 = polygon1.exterior.xy
# ax.fill(x1, y1, alpha=0.5, fc='blue', label='Polygon 1')

# # Рисуем второй полигон
# x2, y2 = polygon2.exterior.xy
# ax.fill(x2, y2, alpha=0.5, fc='green', label='Polygon 2')

# Рисуем объединенный полигон
x_merged, y_merged = merged_polygon.exterior.xy
ax.fill(x_merged, y_merged, alpha=0.5, fc='red', label='Merged Polygon')

# Настройка графика
ax.set_title('Polygon Union Visualization')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()
ax.set_aspect('equal')
ax.grid(True)

# Показываем график
plt.show()