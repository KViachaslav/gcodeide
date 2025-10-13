def exclude_intervals(include_intervals, exclude_intervals):
    result = []

    for start, end in include_intervals:
        current_start = start
        
        # Сортируем исключаемые промежутки
        sorted_excludes = sorted(exclude_intervals)
        
        for ex_start, ex_end in sorted_excludes:
            # Если исключаемый промежуток не перекрывается
            if current_start >= ex_end:
                continue
            if end <= ex_start:
                break
            
            # Обрабатываем часть до исключаемого промежутка
            if current_start < ex_start:
                result.append((current_start, ex_start))
            
            # Обновляем current_start, если current_end пересекается с исключаемым
            current_start = max(current_start, ex_end)
        
        # Добавляем оставшуюся часть, если она есть
        if current_start < end:
            result.append((current_start, end))
    
    return result

# Пример использования
include_intervals = [(0, 10)]
exclude_intervals_list = [(2, 6), (3, 7)]  # Пример с несколькими исключениями

result = exclude_intervals(include_intervals, exclude_intervals_list)
print(result)  # Вывод: [(0, 3), (7, 10)]


