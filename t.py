import matplotlib.pyplot as plt
import numpy as np

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
def find_intersection(x1, y1, x2, y2, c):
    
    if (y1 - c) * (y2 - c) > 0:
        return None

    if y1 == c:
        return (x1, y1)
    if y2 == c:
        return (x2, y2)
    t = (c - y1) / (y2 - y1)


    if 0 <= t <= 1:
        
        x_intersection = x1 + t * (x2 - x1)
        return (x_intersection, c)
    
    return None 
def draw_hatched_area(rect, circles,rectangles):
    x_min, y_min, x_max, y_max = rect
    y_lines = []
    step = 0.09
    for y in np.arange(y_min, y_max, step):
        x_start = x_min
        x_end = x_max
        include_intervals = [(x_start,x_end)]
        exclude_intervals_list = []
        for (x_center, y_center, radius) in circles: 
            if abs(y - y_center) <= radius:  
                delta_x = np.sqrt(radius**2 - (y - y_center)**2)
                x_left = x_center - delta_x
                x_right = x_center + delta_x
                exclude_intervals_list.append((x_left,x_right))
        
        for rect in rectangles:
            p = []
            for i in range(3):
                if find_intersection(rect[i][0],rect[i][1],rect[i+1][0],rect[i+1][1],y):
                    p.append(find_intersection(rect[i][0],rect[i][1],rect[i+1][0],rect[i+1][1],y))
            if len(p) == 2:
                print(p)
                exclude_intervals_list.append((p[0][0],p[1][0]))

        


        result = exclude_intervals(include_intervals, exclude_intervals_list)
        for r in result:
            y_lines.append((r[0],y,r[1],y))   

    
    for (x_start, y, x_end, y) in y_lines:
        plt.plot([x_start, x_end], [y, y], color='black')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal')
    plt.show() 
    return y_lines

rect = (0, 0, 10, 10)
circles = []
rects = [((1,1),(1.2,3.4),(4,3),(4.3,0.6))]
draw_hatched_area(rect, circles,rects)

