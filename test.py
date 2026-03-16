import cv2
import numpy as np

def find_black_edges(image_path):
    # Загружаем изображение в градациях серого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Применяем бинаризацию
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Находим контуры
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Получаем массив координат границ
    edge_points = []
    for contour in contours:
        for point in contour:
            edge_points.append((point[0][0], point[0][1]))
    
    return edge_points

# Замените 'path_to_image.jpg' на путь к вашему изображению
edges_coordinates = find_black_edges('aaa.png')

# Выводим результаты
print(edges_coordinates)