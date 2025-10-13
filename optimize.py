import ezdxf
import svgwrite
import math


def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def find_closest_point(lines, target_point,nums):
    closest_point = None
    min_distance = float('inf')
    I = 0
    i = 0
    mode = 1
    for line in lines:
        if i in nums:
            m = 1
            for point in [line['start'], line['end']]:
                dist = distance(point, target_point)
                if dist < min_distance:
                    min_distance = dist
                    closest_point = point
                    mode = m
                    I = i
                m = 0
            
        i += 1

    return closest_point,I,mode

def dxf_to_svg(dxf_file, svg_file): 

    doc = ezdxf.readfile(dxf_file)
    dwg = svgwrite.Drawing(svg_file, profile='tiny')

    for entity in doc.modelspace().query('LINE'):
        start = entity.dxf.start
        end = entity.dxf.end
        dwg.add(dwg.line(start=(start.x, start.y), end=(end.x, end.y), stroke=svgwrite.rgb(0, 0, 0, '%')))

    dwg.save()

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

def create_continuous_lines(file_path_out, lines):
    doc = ezdxf.new()
    msp = doc.modelspace()

    if not lines:
        return
    nums = set(range(len(lines)))

    p,i,m = find_closest_point(lines,(0,0),range(len(lines)))
    
    if m:
        msp.add_line(lines[i]['start'], lines[i]['end'])
        current_start = lines[i]['end']
    else:
        msp.add_line(lines[i]['end'], lines[i]['start'])
        current_start = lines[i]['start']
    nums.remove(i)
    
    
    while nums:
        p,j,m = find_closest_point(lines,current_start,nums)

        if m:
            msp.add_line(lines[j]['start'], lines[j]['end'])
            current_start = lines[j]['end']
        else:
            msp.add_line(lines[j]['end'], lines[j]['start'])
            current_start = lines[j]['start']

        nums.remove(j)

    doc.saveas(file_path_out)

# input_file = 'out.dxf' 
# output_file = 'Sketch11.dxf'  

# lines = read_dxf_lines(input_file)

# create_continuous_lines(output_file, lines)
# #dxf_to_svg(output_file, '100.svg')
