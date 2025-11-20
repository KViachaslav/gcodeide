import numpy as np
import matplotlib.pyplot as plt

# Определите функцию для радиуса в полярных координатах
def r(theta):
    return 1 + np.sin(3 * theta)  # пример: трепетная звезда

# Генерация углов от 0 до 2π
theta = np.linspace(0, 2 * np.pi, 1000)

# Вычисляем радиус
radius = r(theta)
Roe = 39.7887
Roeo = 0.795775
e = 0.625
# Преобразуем в декартовы координаты
x = (Roe + Roeo) * np.cos(theta) - e * np.cos((Roe+Roeo) * theta/Roeo)
y = (Roe + Roeo) * np.sin(theta) - e * np.sin((Roe+Roeo) * theta/Roeo)

# Построение графика
plt.figure(figsize=(8, 8))
plt.plot(x, y)
plt.title("Кривая в полярных координатах")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")  # Равные масштабы по осям
plt.grid()
plt.show()