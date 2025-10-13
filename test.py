def find_intersection(x1, y1, x2, y2, c):
    # Проверка условий на наличие пересечения
    if (y1 - c) * (y2 - c) > 0:
        return None  # Пересечения нет

    # Если одно из значений y равно c
    if y1 == c:
        return (x1, y1)
    if y2 == c:
        return (x2, y2)

    # Нахождение параметра t
    t = (c - y1) / (y2 - y1)

    # Проверка, находится ли t в пределах отрезка
    if 0 <= t <= 1:
        # Нахождение координат точки пересечения
        x_intersection = x1 + t * (x2 - x1)
        return (x_intersection, c)
    
    return None  # Пересечения нет

# Пример использования
x1, y1 = 1, 1
x2, y2 = 4, 50
c = 3

intersection = find_intersection(x2, y2, x1, y1, c)
if intersection:
    print(f"Точка пересечения: {intersection}")
else:
    print("Пересечение не найдено.")