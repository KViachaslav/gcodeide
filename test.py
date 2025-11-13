import math
import matplotlib.pyplot as plt
import numpy as np

def calculate_gear_parameters(r_delitelni, z_zubiev, alpha_degrees=20):
    """
    Рассчитывает ключевые геометрические параметры эвольвентной шестерни.
    """
    alpha_rad = math.radians(alpha_degrees)
    d = 2 * r_delitelni
    m = d / z_zubiev
    r_b = r_delitelni * math.cos(alpha_rad)
    h_a = m 
    h_f = 1.25 * m
    r_a = r_delitelni + h_a
    r_f = r_delitelni - h_f
    
    def inv_t_np(t):
        return np.tan(t) - t

    try:
        t_a = np.arccos(r_b / r_a) 
    except ValueError:
        t_a = 0 

    results = {
        "r": r_delitelni, "z": z_zubiev, "alpha_rad": alpha_rad, 
        "m": m, "d": d, "r_b": r_b, "r_a": r_a, "r_f": r_f,
        "h_a": h_a, "h_f": h_f, "t_a": t_a, 
        "inv_t": inv_t_np
    }
    return results

def plot_gear_profile(gear_data, num_points=100, num_teeth_to_show=3):
    """
    Визуализирует профиль эвольвентной шестерни с помощью Matplotlib.
    """
    r, z = gear_data['r'], gear_data['z']
    r_b, r_a, r_f = gear_data['r_b'], gear_data['r_a'], gear_data['r_f']
    t_a = gear_data['t_a']
    inv_t = gear_data['inv_t']
    
    # Генерация точек одной стороны эвольвенты
    t_values = np.linspace(0, t_a, num_points)
    
    # Полярный угол для первой стороны эвольвенты
    # Смещение на половину толщины зуба по делительной окружности
    # Отсчет угла начинается от центра зуба
    initial_offset_angle = (math.pi / (2 * z)) # Угол от оси симметрии зуба до начала эвольвенты
    
    rho = r_b / np.cos(t_values)
    phi_one_side = inv_t(t_values) # Угол инволюты
    
    # Координаты первой стороны зуба (относительно центра зуба)
    x_evo_one = rho * np.cos(initial_offset_angle - phi_one_side) # Вычитаем, чтобы двигаться в нужном направлении
    y_evo_one = rho * np.sin(initial_offset_angle - phi_one_side)

    # Вторая сторона зуба - зеркальное отражение первой относительно оси X
    # (если ось X проходит через центр зуба)
    x_evo_other = x_evo_one
    y_evo_other = -y_evo_one # Просто инвертируем Y для симметрии
    
    # Объединяем две стороны, чтобы получить полный профиль зуба
    # Начинаем с верхней точки первой стороны, идем вниз, потом вверх по второй стороне
    x_tooth_profile = np.concatenate([np.flip(x_evo_other), x_evo_one])
    y_tooth_profile = np.concatenate([np.flip(y_evo_other), y_evo_one])
    
    # Теперь, когда у нас есть полный профиль ОДНОГО зуба, мы можем его вращать
    
    pitch_angle = (2 * math.pi) / z # Угловой шаг между центрами зубьев
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Окружности для справки
    angles = np.linspace(0, 2 * np.pi, 360)
    ax.plot(r_a * np.cos(angles), r_a * np.sin(angles), 'r--', linewidth=0.5, label='Вершин ($r_a$)')
    ax.plot(r * np.cos(angles), r * np.sin(angles), 'g--', linewidth=1, label='Делительный ($r$)')
    ax.plot(r_b * np.cos(angles), r_b * np.sin(angles), 'b--', linewidth=0.5, label='Основной ($r_b$)')
    ax.plot(r_f * np.cos(angles), r_f * np.sin(angles), 'k--', linewidth=0.5, label='Впадин ($r_f$)')
    
    # Построение зубьев
    for i in range(z):
        if i >= num_teeth_to_show:
            break
            
        # Угол поворота для текущего зуба
        # Каждый зуб центрируется относительно своей оси
        rotation_angle = i * pitch_angle
        
        # Поворот координат зуба
        c, s = np.cos(rotation_angle), np.sin(rotation_angle)
        R_matrix = np.array([[c, -s], [s, c]])
        
        rotated_x_tooth = x_tooth_profile * c - y_tooth_profile * s
        rotated_y_tooth = x_tooth_profile * s + y_tooth_profile * c
        
        # Профиль зуба
        ax.plot(rotated_x_tooth, rotated_y_tooth, 'k-', linewidth=2)
        ax.fill(rotated_x_tooth, rotated_y_tooth, color='grey', alpha=0.5)
    # Соединение впадин (аппроксимация дугой окружности r_f)
        for i in range(num_teeth_to_show):
            # Углы для дуги впадины между зубами
            # Середина впадины находится ровно посередине между центрами зубьев
            start_angle = i * pitch_angle + half_pitch_angle + initial_offset_angle
            end_angle = (i + 1) * pitch_angle - half_pitch_angle - initial_offset_angle
            
            arc_angles = np.linspace(start_angle, end_angle, 50)
            ax.plot(r_f * np.cos(arc_angles), r_f * np.sin(arc_angles), 'k-', linewidth=2)


        ax.set_title(f'Профиль эвольвентной шестерни (r={r}, z={z})')
        ax.set_xlabel('X, мм')
        ax.set_ylabel('Y, мм')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True)
        ax.legend(loc='upper right')
        plt.savefig('gear_profile_plot.png')
        plt.close()

# --- Пример использования с исходными данными ---
R_del = 50.0  # Делительный радиус (мм)
Z_zub = 20    # Количество зубьев

gear_params = calculate_gear_parameters(R_del, Z_zub)
plot_gear_profile(gear_params, num_teeth_to_show=3)
print("График профиля шестерни успешно сохранен как gear_profile_plot.png")