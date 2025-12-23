from shapely.geometry import LineString,MultiLineString
import shapely

# Создаем несколько линий
line1 = LineString([(0, 0), (1, 1)])
line2 = LineString([(1, 1), (2, 2)])
line3 = LineString([(2, 2), (3, 3)])

# Объединяем линии
merged_line = shapely.line_merge(MultiLineString([line1, line2, line3]))

print(merged_line)