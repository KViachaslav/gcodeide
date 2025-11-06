import sqlite3
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import linemerge

# 1. Настройка и подключение к базе данных
DATABASE_NAME = 'polylines.db'

def merge_polylines(db_name):
    
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

            
        cursor.execute('''
            SELECT polyline_tag, x, y, id 
            FROM coordinates 
            ORDER BY polyline_tag, id
        ''')
        all_data = cursor.fetchall()
        
        # Группировка точек в ломаные линии по polyline_tag
        lines_to_merge = []
        current_tag = None
        current_points = []
        
        for tag, x, y, order in all_data:
            if tag != current_tag and current_points:
                if len(current_points) >= 2:
                    lines_to_merge.append(LineString(current_points))
                current_points = []
            
            current_tag = tag
            current_points.append((x, y))

        if current_points and len(current_points) >= 2:
             lines_to_merge.append(LineString(current_points))
             
        if not lines_to_merge:
            print("Нет ломаных линий для слияния.")
            return []

        # 5. Выполнение слияния
        # linemerge работает с MultiLineString или списком LineStrings
        merged_geometry = linemerge(lines_to_merge)

        # 6. Обработка результата
        results = []
        if merged_geometry.geom_type == 'LineString':
            results.append(merged_geometry)
        elif merged_geometry.geom_type == 'MultiLineString':
            # MultiLineString содержит несколько объединенных ломаных
            results.extend(merged_geometry.geoms)
        
        return results

    except sqlite3.Error as e:
        print(f"Ошибка SQLite: {e}")
        return []
    finally:
        if conn:
            conn.close()

results = merge_polylines(DATABASE_NAME)


for i, line in enumerate(results):
    print(f"Объединенная линия {i+1} ({line.geom_type}):")
    # Преобразование координат в список для вывода
    coords = list(line.coords)
    print(f"  Начало: {coords[0]}")
    print(f"  Конец: {coords[-1]}")
    # print(f"  Все координаты: {coords}") 
    print(f"  Количество вершин: {len(coords)}")