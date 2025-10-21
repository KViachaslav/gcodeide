# !pip install ipywidgets pillow matplotlib
 
import io
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from PIL import Image
from IPython.display import display
 
# Загрузка изображения
upload = widgets.FileUpload(accept='image/*', multiple=False)
display(upload)
 
def on_upload_change(change):
    if upload.value:
        img_data = list(upload.value.values())[0]['content']
        img = Image.open(io.BytesIO(img_data)).resize((500, 500))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
 
upload.observe(on_upload_change, names='value')
 
# Создание холста и переменные
canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255
fig, ax = plt.subplots()
ax.imshow(canvas)
ax.axis('off')
start_point, end_point, tool = None, None, 'line'
 
# Функция для рисования
def draw(event):
    global start_point, end_point
    if event.inaxes:
        if start_point is None:
            start_point = (int(event.xdata), int(event.ydata))
        else:
            end_point = (int(event.xdata), int(event.ydata))
            if tool == 'line':
                ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='black')
            elif tool == 'rectangle':
                ax.add_patch(plt.Rectangle(start_point, end_point[0] - start_point[0], end_point[1] - start_point[1], fill=None, edgecolor='black'))
            elif tool == 'circle':
                radius = ((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)**0.5
                ax.add_patch(plt.Circle(start_point, radius, fill=None, edgecolor='black'))
            start_point, end_point = None, None
            fig.canvas.draw()
 
fig.canvas.mpl_connect('button_press_event', draw)
 
# Виджет для выбора инструмента
tool_selector = widgets.ToggleButtons(options=['line', 'rectangle', 'circle'], description='Tool:')
tool_selector.observe(lambda change: globals().update(tool=change['new']), names='value')
display(tool_selector)
 
plt.show()