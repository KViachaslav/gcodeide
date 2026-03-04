import ezdxf
import matplotlib.pyplot as plt

def extract_text_from_dxf(file_path):
    # Чтение DXF файла
    doc = ezdxf.readfile(file_path)
    texts = []

    # Извлечение текстовых объектов
    for entity in doc.modelspace().query('TEXT'):
        texts.append((entity.dxf.text, entity.dxf.insert))

    return texts

# Замените на путь к вашему DXF файлу
dxf_file = "DXF_PCB1_2026-03-02_AutoCAD2007.dxf"
texts = extract_text_from_dxf(dxf_file)

# Создание графика
fig, ax = plt.subplots()

# Настройка осей
ax.axis('equal')
ax.set_xlim(-10, 100)
ax.set_ylim(-100, 10)

# Отображение текста
for text, insert in texts:
    ax.text(insert.x, insert.y, text, fontsize=12, ha='center', va='center')

plt.title('Текст из DXF файла')
plt.xlabel('X ось')
plt.ylabel('Y ось')
plt.grid()
plt.show()