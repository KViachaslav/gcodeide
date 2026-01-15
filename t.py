import matplotlib.pyplot as plt
import shapely
import numpy as np
from shapely.geometry import Polygon, MultiPoint,LineString, Point, MultiLineString
xx = [56.4004, 56.4929, 56.6108, 56.7493, 56.9033, 57.0667, 57.2333, 57.3967, 57.5507, 57.6892, 57.7113, 59.69, 59.7522, 59.8743, 59.9893, 60.0928, 60.139, 62.493, 62.9387, 62.9608, 63.0993, 63.2533, 63.4167, 63.5833, 63.7467, 63.9007, 64.0392, 64.0613, 68.0187, 68.0408, 68.1793, 68.3333, 68.4967, 68.6633, 68.8267, 68.9807, 69.1192, 69.2371, 69.3296, 69.3934, 69.4259, 69.4259, 69.3934, 69.3296, 69.2371, 69.1192, 68.9807, 68.8267, 68.6633, 68.4967, 68.3333, 68.1793, 68.0408, 68.0187, 64.0613, 64.0392, 63.9007, 63.7467, 63.5833, 63.4167, 63.2533, 63.0993, 62.9608, 62.9387, 62.23, 62.1678, 62.0457, 61.9307, 61.8272, 61.8271, 61.781, 59.427, 57.7113, 57.6892, 57.5507, 57.3967, 57.2333, 57.0667, 56.9033, 56.7493, 56.6108, 56.4929, 56.4004, 56.3366, 56.3041, 56.3041, 56.3366, 56.4004]
yy = [-103.7393, -103.6008, -103.4829, -103.3904, -103.3266, -103.2941, -103.2941, -103.3266, -103.3904, -103.4829, -103.505, -103.505, -103.5081, -103.5323, -103.58, -103.6491, -103.691, -106.045, -106.045, -106.0229, -105.9304, -105.8666, -105.8341, -105.8341, -105.8666, -105.9304, -106.0229, -106.045, -106.045, -106.0229, -105.9304, -105.8666, -105.8341, -105.8341, -105.8666, -105.9304, -106.0229, -106.1408, -106.2793, -106.4333, -106.5967, -106.7633, -106.9267, -107.0807, -107.2192, -107.3371, -107.4296, -107.4934, -107.5259, -107.5259, -107.4934, -107.4296, -107.3371, -107.315, -107.315, -107.3371, -107.4296, -107.4934, -107.5259, -107.5259, -107.4934, -107.4296, -107.3371, -107.315, -107.315, -107.3119, -107.2877, -107.24, -107.1709, -107.1709, -107.129, -104.775, -104.775, -104.7971, -104.8896, -104.9534, -104.9859, -104.9859, -104.9534, -104.8896, -104.7971, -104.6792, -104.5407, -104.3867, -104.2233, -104.0567, -103.8933, -103.7393]
nls = []
def expand_polygon(original_polygon, distance):
    """Расширяет полигон на заданное расстояние."""
    
    # Проверка, что входной объект является полигон
    if not isinstance(original_polygon, Polygon):
        raise ValueError("Входной параметр должен быть экземпляром Polygon.")
    
    # Получаем координаты вершин полигона
    exterior = original_polygon.exterior
    points = list(exterior.coords)
    
    # Список для хранения новых координат
    new_points = []
    
    # Проходим по каждому отрезку полигона
    for i in range(len(points) - 1):
        p1 = Point(points[i])
        p2 = Point(points[i + 1])
        
        # Вычисляем направление отрезка
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        
        # Вычисляем длину отрезка
        length = (dx**2 + dy**2) ** 0.5
        
        # Нормализуем вектор отрезка
        if length == 0:
            continue  # Игнорируем нулевой отрезок
        unit_dx = dx / length
        unit_dy = dy / length
        
        # Находим перпендикулярный вектор
        perp_dx = -unit_dy
        perp_dy = unit_dx
        
        # Вычисляем новые координаты концов отрезка
        new_p1 = (p1.x + perp_dx * distance, p1.y + perp_dy * distance)
        new_p2 = (p2.x + perp_dx * distance, p2.y + perp_dy * distance)
        nls.append(LineString([[p1.x + perp_dx * distance, p1.y + perp_dy * distance], [p2.x + perp_dx * distance, p2.y + perp_dy * distance]]))
        new_points.append(new_p1)
    
    # Замыкаем новый полигон
    new_points.append(new_points[0])  # Замыкаем на первый пункт
    new_polygon = Polygon(new_points)
    
    return new_polygon


def get_radius(a,b,x,y):
    a = 0.2
    b = 0.04
    if y == 0:
        return b
    if x == 0:
        return a
    else:
        ans = a*b/np.sqrt((b*np.cos(np.arctan(x/y)))**2 + (a*np.sin(np.arctan(x/y)))**2) 
        return ans + (a-ans) * 0.5

polygon = Polygon([(x,y)for x,y in zip(xx,yy)])
npol = expand_polygon(polygon,0.3)
npol = shapely.concave_hull(MultiLineString(nls), ratio=0.14)
# # Получаем координаты внешней границы
points = np.array(polygon.exterior.coords)
pr = np.array(npol.exterior.coords)

# # Функция для нахождения касательной в точке
# def tangent_at_point(point1, point2):
#     return (point2[0] - point1[0], point2[1] - point1[1])
# pp = []
# # Проход по всем точкам полигона для нахождения касательных и перпендикуляров
# tangents = []
# perpendiculars = []
# length = 0.5  # Длина перпендикуляра

# for i in range(len(points) - 1):
#     p1 = points[i]
#     p2 = points[i + 1]
#     tangent = tangent_at_point(p1, p2)
#     tangents.append((p1, tangent))
    
#     # Вычисляем перпендикулярный вектор
#     perp_vector = (-tangent[1], tangent[0])
    
#     # angle_radians = np.arctan2(perp_vector[1], perp_vector[0])

#     # angle_degrees = np.degrees(angle_radians)
    


#     # Нормализация перпендикуляра
#     norm = np.sqrt(perp_vector[0]**2 + perp_vector[1]**2)
#     perp_vector = (perp_vector[0] / norm * get_radius(0.2,0.2,perp_vector[1],perp_vector[0]), perp_vector[1] / norm * get_radius(0.2,0.2,perp_vector[1],perp_vector[0]))
    
#     # Конечная точка перпендикуляра
#     end_point = (p1[0] + perp_vector[0], p1[1] + perp_vector[1])
#     pp.append(end_point)
#     perpendiculars.append((p1, end_point))



plt.plot(pr[:, 0], pr[:, 1])

# # # Рисуем полигон
plt.plot(points[:, 0], points[:, 1], 'b-', label='Polygon')


# for start, end in perpendiculars:
#     plt.plot([start[0], end[0]], [start[1], end[1]])

# Делаем график красивым
plt.xlim(55,70)
plt.ylim(-100,-110)
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.grid()
plt.gca().set_aspect('equal', adjustable='box')
# plt.legend()
plt.title('Polygon, Tangents and Perpendiculars')
plt.show()