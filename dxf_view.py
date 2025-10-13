import ezdxf
import matplotlib.pyplot as plt
## Показывает путь гравировки(резки): черный - гравировка; серый - холостое перемещение 




def read_dxf_lines(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    lines = []

    for line in msp.query('LINE'):
        lines.append({
            'start': (line.dxf.start.x, line.dxf.start.y),
            'end': (line.dxf.end.x, line.dxf.end.y)
        })

    return lines

def plot_lines(lines):
    plt.figure(figsize=(8, 8))
    temp_point = (0,0)
    for line in lines:
        start = line['start']
        end = line['end']
        if abs(temp_point[0] - start[0])> 0.01 or abs(temp_point[1] - start[1])> 0.01:
            plt.plot([temp_point[0], start[0]], [temp_point[1], start[1]],color= (0.9,0.9,0.9))
        plt.plot([start[0], end[0]], [start[1], end[1]],color= (0,0,0))
        temp_point = end
   
    plt.title('Lines from DXF file')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.axis('equal')  
    plt.show()
    plt.close()


file_path = 'out.dxf'
lines = read_dxf_lines(file_path)
print(len(lines))
plot_lines(lines)