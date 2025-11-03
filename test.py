import numpy as np
import matplotlib.pyplot as plt

def bezier_curve(p0, p1, p2, num_points=100):
    t = np.linspace(0, 1, num_points)
    curve = (1 - t)[:, np.newaxis]**2 * p0 + 2 * (1 - t)[:, np.newaxis] * t[:, np.newaxis] * p1 + t[:, np.newaxis]**2 * p2
    return curve

# Определяем контрольные точки
p0 = np.array([0, 0])
p1 = np.array([1, 2])
p2 = np.array([2, 0])

# Генерируем кривую
curve = bezier_curve(p0, p1, p2)

# Визуализируем
plt.plot(curve[:, 0], curve[:, 1], label='Кривая Безье')
plt.plot([p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]], 'ro--', label='Контрольные точки')
plt.legend()
plt.title('Кривая Безье')
plt.grid()
plt.axis('equal')
plt.show()