from shapely.geometry import box

# Создаем основной прямоугольник (x_min, y_min, x_max, y_max)
main_rectangle = box(0, 0, 10, 10)

# Список прямоугольников для вычитания
subtractions = [
    box(1, 1, 4, 4),
    box(5, 5, 7, 7),
    box(3, 3, 6, 6)
]

# Вычитание каждого прямоугольника из основного
for subtraction in subtractions:
    main_rectangle = main_rectangle.difference(subtraction)

# Площадь оставшейся фигуры
remaining_area = main_rectangle.exterior.xy

print(f"Остаточная площадь после вычитания: {remaining_area}")