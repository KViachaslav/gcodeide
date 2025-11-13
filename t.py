from shapely.geometry import LineString, Point, GeometryCollection
from shapely.ops import split

def split_linestring_at_nearest_point(line_coords, target_point_coords, tolerance=1e-9):
    
    line = LineString(line_coords)
    target_point = Point(target_point_coords)

    distance_along_line = line.project(target_point)

    split_point = line.interpolate(distance_along_line)

    if split_point.equals_exact(Point(line.coords[0])) or split_point.equals_exact(Point(line.coords[-1])):
        print("Ближайшая точка совпадает с началом или концом ломаной. Разделение не требуется.")
        return [line]

    result = split(line, split_point)

    if isinstance(result, GeometryCollection):
        split_lines = [geom for geom in result.geoms if isinstance(geom, LineString)]
        if len(split_lines) == 2:
            return split_lines
        else:
        
            print("split() вернул неожиданный результат. Использование ручного обрезания.")
            return cut_line(line, distance_along_line)
    
    return [line]


def cut_line(line, distance):
    """Вспомогательная функция для ручного разрезания ломаной по расстоянию."""
    if distance <= 0.0 or distance >= line.length:
        return [line]

    coords = list(line.coords)
    for i in range(len(coords) - 1):
        p1 = Point(coords[i])
        p2 = Point(coords[i+1])
        segment = LineString([p1, p2])
        segment_len = segment.length
        
        current_distance = sum(LineString(coords[j:j+2]).length for j in range(i)) + segment_len
        
        if current_distance >= distance:
           
            
            dist_on_segment = distance - (current_distance - segment_len)
            
            split_point = segment.interpolate(dist_on_segment)
            
            coords1 = coords[:i+1] + [split_point.coords[0]]
            
            coords2 = [split_point.coords[0]] + coords[i+1:]
            
            return [LineString(coords1), LineString(coords2)]

    return [line] 


# Исходная ломаная (массив с координатами точек)
line_coordinates = [(0, 0), (5, 5), (10, 0), (15, 5)]

# Заданная точка, к которой ищется ближайшая
target_point_coordinates = (8, 6) 

# Выполнение разделения
polyline_segments = split_linestring_at_nearest_point(line_coordinates, target_point_coordinates)
## Вывод результата
if polyline_segments and len(polyline_segments) == 2:
    print(f"Исходная ломаная: {LineString(line_coordinates).wkt}")
    print(f"Заданная точка: {Point(target_point_coordinates).wkt}")
    print("\n✅ Ломаная разделена на 2 части:")
    print(f"1-я часть: {polyline_segments[0].wkt}")
    print(f"2-я часть: {polyline_segments[1].wkt}")
else:
    print("\n⚠️ Разделение не выполнено или результат не содержит 2 частей LineString.")

# Вывод ближайшей точки
line_obj = LineString(line_coordinates)
distance_on_line = line_obj.project(Point(target_point_coordinates))
nearest_p = line_obj.interpolate(distance_on_line)
print(f"\nБлижайшая точка на ломаной: {nearest_p.wkt}")
