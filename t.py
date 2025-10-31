import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

def extract_points_from_ggb(ggb_file_path):
   
    points = {}
    segments = []  # Список сегментов: [(точка1_label, точка2_label, seg_label)]
    
    # Шаг 1: Распаковка ZIP и чтение XML
    try:
        with zipfile.ZipFile(ggb_file_path, 'r') as zip_ref:
            xml_content = zip_ref.read('geogebra.xml').decode('utf-8')
            
    except Exception as e:
        print(f"Ошибка при чтении ZIP: {e}")
        return points, segments
    
    # Шаг 2: Парсинг XML
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Ошибка парсинга XML: {e}")
        return points, segments
    
    # Шаг 3: Извлечение точек (<element type="point">)
    for elem in root.iter('element'):
        if elem.get('type') == 'point':
            label = elem.get('label', 'unnamed')
            coords_elem = elem.find('coords')  # <coords x="..." y="..." z="..."/>
            if coords_elem is not None:
                x = float(coords_elem.get('x', 0.0))
                y = float(coords_elem.get('y', 0.0))
                points[label] = (x, y)
    
    # Шаг 4: Извлечение сегментов (<command name="Segment">)
    for command in root.iter('command'):
        cmd_name = command.get('name')
        
        if cmd_name == 'Segment':
            input_elem = command.find('input')
            if input_elem is not None:
                a0 = input_elem.get('a0')  # Первая точка
                a1 = input_elem.get('a1')  # Вторая точка
                output = command.find('output')
                seg_label = output.get('a0') if output is not None else f"seg_{len(segments)}"
                
                if a0 in points and a1 in points:
                    segments.append((points[a0], points[a1]))
                    
        
        elif cmd_name == 'Polygon' or  cmd_name == 'PolyLine':
           
            input_elem = command.find('input')
            output = command.find('output')
            poly_label = output.get('a0') if output is not None else f"poly_{len(segments)}"
            
            if input_elem is not None:
                # Извлекаем все точки вершин (a0, a1, a2, ... — переменное число)
                vertices = []
                i = 0
                while True:
                    vertex_label = input_elem.get(f'a{i}')
                    if vertex_label is None:
                        break
                    if vertex_label in points:
                        vertices.append(vertex_label)
                    else:
                        print(f"Предупреждение: Полigon '{poly_label}' ссылается на неизвестную точку '{vertex_label}'")
                    i += 1
                
                  
                    
                    
                for j in range(len(vertices)):
                    start_vertex = vertices[j]
                    end_vertex = vertices[(j + 1) % len(vertices)] 
                    
                    seg_label = f"{poly_label}_side{j+1}"
                    if output is not None and j < len([k for k in output.attrib if k.startswith('a')]) - 1:
                        seg_label = output.get(f'a{j+1}', seg_label)
                    
                    segments.append((points[start_vertex], points[end_vertex]))
                    
    
    return points, segments

if __name__ == "__main__":
    ggb_path = Path("material-sa25ebtp (2).ggb")  # Замените на путь к вашему .ggb!
    if ggb_path.exists():
        points, segments = extract_points_from_ggb(ggb_path)
        print(f"\nВсего точек: {len(points)}")
        print(f"Всего сегментов: {len(segments)}")
        print(segments)
    