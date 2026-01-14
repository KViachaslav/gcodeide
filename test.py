from PIL import Image
import numpy as np

# Открываем изображение
image_path = 'dig (10).png'  # Укажите путь к вашему изображению

image = Image.open(image_path)

# Проверяем размер изображения
if image.size != (32, 50):
    raise ValueError("Изображение должно быть размером 32x50")

# Преобразуем изображение в черно-белое
image_bw = image.convert('1')  # '1' - черно-белый формат

# Преобразуем изображение в массив пикселей
image_data = np.array(image_bw)

# Создаем массив для хранения uint32
uint32_array = np.zeros((50,), dtype=np.uint32)

# Заполняем uint32_array
for row in range(50):
    value = 0
    for column in range(32):
        # Устанавливаем соответствующий бит (0 - черный, 1 - белый)
       
        if image_data[row, column] == 1:  # Белый пиксель
            value |= (1 << column)
    uint32_array[row] = value

# Печатаем результат

st = ""
for i, val in enumerate(uint32_array):
    st += f"0x{val:08x}, "

print(st)