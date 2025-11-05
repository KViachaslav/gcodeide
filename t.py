import dxfgrabber
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, splprep, splev # Импортируем обе функции

def plot_dxf_spline_fixed(dxf_file_path):
    
    try:
        dwg = dxfgrabber.readfile(dxf_file_path)
    except FileNotFoundError:
        print(f"Ошибка: Файл '{dxf_file_path}' не найден.")
        return
    
    
    spline_count = 0
    
    for entity in dwg.entities:
        if entity.dxftype == 'SPLINE':
            spline_count += 1
            
            degree = entity.degree
            
           
            if hasattr(entity, 'knots') and entity.knots:
                
                knot_vector = np.array(entity.knots)
                control_points = np.array(entity.control_points)
                
                x = control_points[:, 0]
                y = control_points[:, 1]
                
                
                weights = getattr(entity, 'weights', None) 
                if weights is not None and not all(w == 1.0 for w in weights):
                    print(f"Предупреждение: Сплайн {spline_count} является NURBS. Используется нерациональное приближение.")
                
                try:
                    
                    u_min = knot_vector[degree]
                    u_max = knot_vector[len(knot_vector) - degree - 1]
                    u_range = np.linspace(u_min, u_max, 100)
                    
                    spline_x = BSpline(knot_vector, x, degree)
                    spline_y = BSpline(knot_vector, y, degree)
                    
                    curve_x = spline_x(u_range)
                    curve_y = spline_y(u_range)
                    
                    ax.plot(curve_x, curve_y, '-', label=f'Сплайн {spline_count} (BSpline)')
                    
                    
                except Exception as e:
                    print(f"Ошибка SciPy при обработке BSpline {spline_count}: {e}")

            elif hasattr(entity, 'fit_points') and entity.fit_points:
                
                fit_points = np.array(entity.fit_points)
                x = fit_points[:, 0]
                y = fit_points[:, 1]
                
                try:
                    # Используем splprep для автоматической генерации узлов и коэффициентов
                    tck, u = splprep([x, y], k=degree, s=0) 
                    u_new = np.linspace(u.min(), u.max(), 100)
                    curve_points = splev(u_new, tck)
                    
                    # Отрисовка
                    ax.plot(curve_points[0], curve_points[1], '--', label=f'Сплайн {spline_count} (Интерполяция)')
                    
                except Exception as e:
                    print(f"Ошибка SciPy при интерполяции (Fit Points) {spline_count}: {e}")

            else:
                print(f"Сплайн {spline_count} пропущен: Нет ни knots, ни fit_points.")


    

plot_dxf_spline_fixed("puuppu.dxf")