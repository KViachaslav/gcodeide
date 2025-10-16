import dearpygui.dearpygui as dpg
from fdialog import FileDialog
import optimize

import dearpygui.dearpygui as dpg
import ezdxf
import numpy as np
import math
import svgwrite
from db import SQLiteDatabase
from PIL import Image
import os
from ezdxf import colors
import shapely
from shapely.geometry import Point, LineString, MultiPoint,Polygon,MultiPolygon,MultiLineString
from shapely.ops import unary_union
import re


def active_but(sender,app_data):
    
    state = db.get_records_where('lines',f"parent='{sender}'")[0][7]
    
    db.update_multiple_fields('lines','isactive','parent',sender)
    dpg.bind_item_theme(sender, enabled_theme if state else disabled_theme)
    redraw()

def arc_to_lines(center, radius, start_angle, end_angle, num_segments):
    start_angle_rad = np.radians(start_angle)
    end_angle_rad = np.radians(end_angle)

    angles = np.linspace(start_angle_rad, end_angle_rad, num_segments + 1)

    points = [(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in angles]
    return points
def extract_black_lines(image_path, pixel_distance):
    nice_path = os.path.basename(image_path)
    iter = 1
    while 1:
        for i in db.get_unique_values('lines','parent'):
            if i == nice_path:
                nice_path = os.path.basename(image_path) + f' (copy {iter})'
                iter +=1
        else: 
            break
    dpg.add_button(label=nice_path,parent='butonss',tag=nice_path,callback=active_but)
    
    img = Image.open(image_path).convert('L') 
    img.save("output_bw.png")
    img_array = np.array(img)
    height, width = img_array.shape
    liness = []
    
    for y in range(height):
        start = None
        for x in range(0, width):
            # print(img_array[y, x])
            if img_array[y, x] < 128:  
                if start is None:
                    start = (x, y) 
            else:
                if start is not None:
                    liness.append((round(start[0]*pixel_distance,4),round(start[1]*pixel_distance,4),round(x*pixel_distance,4),round(y*pixel_distance,4),0,nice_path,0,1))
                    
                    start = None
        
        #if start is not None:
            #liness.append((start,y, width - pixel_distance,y,0,nice_path,0,1))

    return liness


def calculate_boundary_coordinates(x1, y1, x2, y2, width):

    dx = x2 - x1
    dy = y2 - y1
    d = math.sqrt(dx**2 + dy**2)
    ux = dx / d
    uy = dy / d

    nx, ny = -uy, ux 
    
    half_width = width / 2
    left_start = (x1 + half_width * nx, y1 + half_width * ny)
    right_start = (x1 - half_width * nx, y1 - half_width * ny)
    left_end = (x2 + half_width * nx, y2 + half_width * ny)
    right_end = (x2 - half_width * nx, y2 - half_width * ny)
    return {
        "left_start": left_start,
        "right_start": right_start,
        "left_end": left_end,
        "right_end": right_end
    }


def exclude_intervals(include_intervals, exclude_intervals):
    result = []
    for start, end in include_intervals:
        current_start = start
        sorted_excludes = sorted(exclude_intervals)
        for ex_start, ex_end in sorted_excludes:
            if current_start >= ex_end:
                continue
            if end <= ex_start:
                break
            if current_start < ex_start:
                result.append((current_start, ex_start))
            current_start = max(current_start, ex_end)
        if current_start < end:
            result.append((current_start, end))
    
    return result

def draw_hatched_area(rect, circles,rectangles):
    x_min, y_min, x_max, y_max = rect
    y_lines = []
    step = 0.1
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
            if find_intersection(rect[0][0],rect[0][1],rect[3][0],rect[3][1],y):
                    p.append(find_intersection(rect[0][0],rect[0][1],rect[3][0],rect[3][1],y))
            if len(p) == 2:
                
                if p[0][0] < p[1][0]:
                    exclude_intervals_list.append((p[0][0],p[1][0]))
                else:
                    exclude_intervals_list.append((p[1][0],p[0][0]))

        


        result = exclude_intervals(include_intervals, exclude_intervals_list)
        for r in result:
            y_lines.append((r[0],y,r[1],y))   

    return y_lines
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
def extend_line(a, b, w):
    
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    
    length = math.sqrt(dx**2 + dy**2)
    
    if length == 0:
        return b 
    
    dx /= length
    dy /= length
    
    c_x = b[0] + dx * w
    c_y = b[1] + dy * w
    
    return c_x, c_y

def points_on_rectangle_sides(corners, distance):
    points = []
    
    sides = [
        LineString([corners[0], corners[1]]),
        LineString([corners[1], corners[2]]),
        LineString([corners[2], corners[3]]),
        LineString([corners[3], corners[0]])  
    ]

    for side in sides:
        length = side.length
       
        num_points = int(length // distance)
        
        for i in range(num_points + 1): 
            point = side.interpolate(i * distance)
            points.append(point)
    return points
def manual_clustering(multipoint, min_distance=2):
   
    points = list(multipoint.geoms)
    clusters = []

    for point in points:
        
        can_add_to_cluster = False
        for cluster in clusters:
            
            for cluster_point in cluster:
                if point.distance(cluster_point) < min_distance:
                    cluster.append(point)
                    can_add_to_cluster = True
                    break
            if can_add_to_cluster:
                break
        
        if not can_add_to_cluster:
            clusters.append([point])
    return [MultiPoint(cluster) for cluster in clusters]
def Polygon_to_lines(union_polygon,num_lines,width_lines,nice_path):
    for k in range(num_lines):

        tunion_polygon = []
        if union_polygon.geom_type == 'Polygon':
            tunion_polygon.append(union_polygon.buffer(width_lines,quad_segs=0))
            xm, ym = union_polygon.exterior.xy
            lins = []
            lines = []
            for i in range(len(xm)-1):
                lins.append((xm[i],ym[i],xm[i+1],ym[i+1]))
            lins.append((xm[len(xm)-1],ym[len(xm)-1],xm[0],ym[0]))

            
            for inter in union_polygon.interiors:
                    xm, ym = inter.xy
                    for i in range(len(xm)-1):
                        lins.append((xm[i],ym[i],xm[i+1],ym[i+1]))
                    lins.append((xm[len(xm)-1],ym[len(xm)-1],xm[0],ym[0]))

            for l in lins:
                lines.append((round(l[0],4),  round(l[1],4), round(l[2],4), round(l[3],4),0,nice_path ,0,1))
            print(len(lines))
            db.add_multiple_records('lines',lines)
        else:
            
            lins = []
            lines = []
            

            for p in union_polygon.geoms:
                
                tunion_polygon.append(p.buffer(width_lines,quad_segs=0))
                
                xm, ym = p.exterior.xy
                
                for i in range(len(xm)-1):
                    lins.append((xm[i],ym[i],xm[i+1],ym[i+1]))
                lins.append((xm[len(xm)-1],ym[len(xm)-1],xm[0],ym[0]))
                
                for inter in p.interiors:
                    xm, ym = inter.xy
                    
                    for i in range(len(xm)-1):
                        lins.append((xm[i],ym[i],xm[i+1],ym[i+1]))
                    lins.append((xm[len(xm)-1],ym[len(xm)-1],xm[0],ym[0]))
                    
            for l in lins:
                lines.append((round(l[0],4),  round(l[1],4), round(l[2],4), round(l[3],4),0,nice_path ,0,1))
            print(len(lines))
            db.add_multiple_records('lines',lines)
        union_polygon = unary_union(MultiPolygon([p for p in tunion_polygon]))
            
def read_dxf_lines_from_esyeda(sender, app_data, user_data):
    doc = ezdxf.readfile(user_data[0])
    full = dpg.get_value('varradio') == 'full'
   
    layers = []
    for i in range(1,len(user_data)):
        if dpg.get_value(user_data[i]):
            layers.append(user_data[i])
        dpg.delete_item(user_data[i])
    dpg.delete_item('CANCEL')
    dpg.delete_item('OK')
    dpg.delete_item('hor_grouph')
    dpg.delete_item('varradio')
    
    dpg.configure_item("modal_id", show=False)
    nice_path = os.path.basename(user_data[0])
    iter = 1
    while 1:
        for i in db.get_unique_values('lines','parent'):
            if i == nice_path:
                nice_path = os.path.basename(user_data[0]) + f' (copy {iter})'
                iter +=1
        else:
            break
    dpg.add_button(label=nice_path ,parent='butonss',tag=nice_path ,callback=active_but)
    doc = ezdxf.readfile(user_data[0])
    msp = doc.modelspace()
    
    num_lines = int(dpg.get_value('border_line_count'))
    width_lines = float(dpg.get_value('border_line_width'))

    border = []
    polygons = []
    for circle in msp.query('CIRCLE'):
        layer = circle.dxf.layer
        
        if layer in layers:
            center = circle.dxf.center    
            num_points = 10  
            radius = circle.dxf.radius + width_lines/2
            polygons.append(Polygon([(center.x + radius * math.cos(2 * math.pi * i / num_points),center.y + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))

           
        
    for polyline in msp.query('LWPOLYLINE'):
        layer = polyline.dxf.layer
        if layer in layers:
            w = polyline.dxf.const_width
            points = polyline.get_points()  
        
            num_points = 10
            radius = w/2 + width_lines/2
            polygons.append(Polygon([(points[0][0] + radius * math.cos(2 * math.pi * i / num_points),points[0][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))    
            for j in range(len(points) - 1):
                num_points = 10
                radius = w/2 + width_lines/2
                boundaries = calculate_boundary_coordinates(points[j][0], points[j][1], points[j + 1][0], points[j + 1][1], w + width_lines)
                polygons.append(Polygon([(points[j + 1][0] + radius * math.cos(2 * math.pi * i / num_points),points[j + 1][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))
                
                polygons.append(Polygon([(boundaries['left_start'][0],boundaries['left_start'][1]), (boundaries['left_end'][0],boundaries['left_end'][1]), (boundaries['right_end'][0],boundaries['right_end'][1]), (boundaries['right_start'][0],boundaries['right_start'][1])]))

        if layer == 'BoardOutLine' and full:
            w = polyline.dxf.const_width
            points = polyline.get_points()
            num_points = 10
            radius = w/2 + width_lines/2
            border.append(Polygon([(points[0][0] + radius * math.cos(2 * math.pi * i / num_points),points[0][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))  
            for j in range(len(points) - 1):
                num_points = 20
                radius = w/2 + width_lines/2
                boundaries = calculate_boundary_coordinates(points[j][0], points[j][1], points[j + 1][0], points[j + 1][1], w + width_lines)
                border.append(Polygon([(points[j + 1][0] + radius * math.cos(2 * math.pi * i / num_points),points[j + 1][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))
                
                border.append(Polygon([(boundaries['left_start'][0],boundaries['left_start'][1]), (boundaries['left_end'][0],boundaries['left_end'][1]), (boundaries['right_end'][0],boundaries['right_end'][1]), (boundaries['right_start'][0],boundaries['right_start'][1])]))





    for hatch in msp.query('HATCH'):
        layer = hatch.dxf.layer
        if layer in layers:
            for path in hatch.paths:
                points = path.vertices
                if len(points) == 4:
                    polygons.append(Polygon([(points[0][0],points[0][1]), (points[1][0],points[1][1]), (points[2][0],points[2][1]), (points[3][0],points[3][1])]).buffer(width_lines/2,quad_segs=2))

    lins = []
    if full:
       
        xm, ym = shapely.envelope(unary_union(MultiPolygon([p for p in border]))).exterior.xy
        xmin = min(xm)
        xmax = max(xm)
        ymin = min(ym)
        ymax = max(ym)
        lins = MultiLineString([((xmin, y), (xmax, y))for y in np.arange(ymin,ymax,width_lines)])

        linn = lins.difference(unary_union(MultiPolygon([p for p in polygons])))
        
        lines = []
        for l in linn.geoms:
            lines.append((round(l.coords[0][0],4),  round(l.coords[0][1],4), round(l.coords[1][0],4), round(l.coords[1][1],4),0,nice_path,0,1))
        db.add_multiple_records('lines',lines)
        Polygon_to_lines(unary_union(MultiPolygon([p for p in polygons])),1,width_lines,nice_path+ '_border')
        dpg.add_button(label=nice_path + '_border',parent='butonss',tag=nice_path + '_border',callback=active_but)
    else:
        multipolygon = MultiPolygon([p for p in polygons])
        union_polygon = unary_union(multipolygon)
        Polygon_to_lines(union_polygon,num_lines,width_lines,nice_path)
        
    redraw()













def read_dxf_lines(file_path):
    nice_path = os.path.basename(file_path)
    iter = 1
    while 1:
        for i in db.get_unique_values('lines','parent'):
            if i == nice_path:
                nice_path = os.path.basename(file_path) + f' (copy {iter})'
                iter +=1
        else: 
            break
    dpg.add_button(label=nice_path,parent='butonss',tag=nice_path,callback=active_but)
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()


    layers = doc.layers
    ll = []
    lll = {}
    pattern = r'^power(\d+)speed(\d+)$'
    h = 1
    for layer in layers:
        match = re.match(pattern, layer.dxf.name)
        if match:
            ll.append(layer.dxf.name)
            power = int(match.group(1))
            speed = int(match.group(2))
            dpg.set_value(f"{h}1_value",power)
            dpg.set_value(f"{h}_value",speed)
            lll[layer.dxf.name] = h - 1 
            h+=1


    lines = []
    for line in msp.query('LINE'):
        layer = line.dxf.layer
        
        if layer in ll:
            
            lines.append((round(line.dxf.start.x,4),  round(line.dxf.start.y,4), round(line.dxf.end.x,4), round(line.dxf.end.y,4),lll[layer],nice_path,0,1))
        else:
            lines.append((round(line.dxf.start.x,4),  round(line.dxf.start.y,4), round(line.dxf.end.x,4), round(line.dxf.end.y,4),0,nice_path,0,1))

        
    hlines = []
    hcol = []
    for line in msp.query('3DFACE'): 
        layer = line.dxf.layer
        if layer in ll:
            hcol.append(lll[layer])
            hcol.append(lll[layer])
            hcol.append(lll[layer])
        else:
            hcol.append(0)
            hcol.append(0)
            hcol.append(0)
        hlines.append({
            'start': (line.dxf.vtx0[0], line.dxf.vtx0[1]),
            'end': (line.dxf.vtx1[0], line.dxf.vtx1[1])
        })
        
        hlines.append({
            'start': (line.dxf.vtx1[0], line.dxf.vtx1[1]),
            'end': (line.dxf.vtx2[0], line.dxf.vtx2[1])
        })
        
        hlines.append({
            'start': (line.dxf.vtx2[0], line.dxf.vtx2[1]),
            'end': (line.dxf.vtx0[0], line.dxf.vtx0[1])
        })
        
    sett = {i for i in range(len(hlines))}
    settt = []
    while sett:
        found = False
        for i in sett:
            for j in sett:
                if i != j:
                    if (abs(hlines[i]['start'][0] - hlines[j]['start'][0])<0.0001 and abs(hlines[i]['start'][1] - hlines[j]['start'][1])<0.0001 and abs(hlines[i]['end'][0] - hlines[j]['end'][0])<0.0001 and abs(hlines[i]['end'][1] - hlines[j]['end'][1])<0.0001) or (abs(hlines[i]['start'][0] - hlines[j]['end'][0])<0.0001 and abs(hlines[i]['start'][1] - hlines[j]['end'][1])<0.0001 and abs(hlines[j]['start'][0] - hlines[i]['end'][0])<0.0001 and abs(hlines[j]['start'][1] - hlines[i]['end'][1])<0.0001):
                        sett.remove(i)
                        sett.remove(j)
                        found = True
                        break
            if found:
                break
            settt.append(i)
            sett.remove(i)
            break
    
    for i in settt:
        lines.append((round(hlines[i]['start'][0],4),  round(hlines[i]['start'][1],4), round(hlines[i]['end'][0],4), round(hlines[i]['end'][1],4),hcol[i],nice_path,0,1))
       
    for acdb_line in msp.query('AcDbLine'):
        layer = acdb_line.dxf.layer
        if layer in ll:
            lines.append((round(acdb_line.dxf.start.x,4),  round(acdb_line.dxf.start.y,4), round(acdb_line.dxf.end.x,4), round(acdb_line.dxf.end.y,4),lll[layer],nice_path,0,1))
        else:
            lines.append((round(acdb_line.dxf.start.x,4),  round(acdb_line.dxf.start.y,4), round(acdb_line.dxf.end.x,4), round(acdb_line.dxf.end.y,4),0,nice_path,0,1))
       
    # for arc in msp.query('ARC'):
    #     center = arc.dxf.center  
    #     radius = arc.dxf.radius   
    #     start_angle = arc.dxf.start_angle 
    #     end_angle = arc.dxf.end_angle
    #     if radius<10:
    #         points = arc_to_lines(center, radius, start_angle, end_angle,10)
    #     else:
    #         points = arc_to_lines(center, radius, start_angle, end_angle,50)
    #     for i in range(len(points)-1):
    #         lines.append((round(points[i][0],4),  round(points[i][1],4), round(points[i+1][0],4), round(points[i+1][1],4),0,nice_path,0,1))
            
    # for circle in msp.query('CIRCLE'):
    #     center = circle.dxf.center 
    #     radius = circle.dxf.radius  
    #     num_points = 50  

    #     points = [
    #         (
    #             center.x + radius * math.cos(2 * math.pi * i / num_points),
    #             center.y + radius * math.sin(2 * math.pi * i / num_points)
    #         )
    #         for i in range(num_points)
    #     ]
        
    #     for i in range(len(points)-1):
    #         lines.append((round(points[i][0],4),  round(points[i][1],4), round(points[i+1][0],4), round(points[i+1][1],4),0,nice_path,0,1))
          

    for polyline in msp.query('SOLID'):
        layer = polyline.dxf.layer
        
        points = polyline.get_points() 
        if layer in ll:
         
            for i in range(len(points) - 1):
                lines.append((round(points[i][0],4),  round(points[i][1],4), round(points[i+1][0],4), round(points[i+1][1],4),lll[layer],nice_path,0,1))
        else:
            for i in range(len(points) - 1):
                lines.append((round(points[i][0],4),  round(points[i][1],4), round(points[i+1][0],4), round(points[i+1][1],4),0,nice_path,0,1))
            

    for polyline in msp.query('LWPOLYLINE'):
        layer = polyline.dxf.layer
        points = polyline.get_points() 
        
        if layer in ll:
            for i in range(len(points) - 1):
                lines.append((round(points[i][0],4),  round(points[i][1],4), round(points[i+1][0],4), round(points[i+1][1],4),lll[layer],nice_path,0,1))
        else:
            for i in range(len(points) - 1):
                lines.append((round(points[i][0],4),  round(points[i][1],4), round(points[i+1][0],4), round(points[i+1][1],4),0,nice_path,0,1))
           
    for hatch in msp.query('HATCH'):
        for path in hatch.paths:
            layer = polyline.dxf.layer
            
            points = path.vertices
            if layer in ll:
                for i in range(len(points)-1):
                    lines.append((round(points[i][0],4),  round(points[i][1],4), round(points[i+1][0],4), round(points[i+1][1],4),lll[layer],nice_path,0,1))
                lines.append((round(points[0][0],4),  round(points[0][1],4), round(points[len(points)-1][0],4), round(points[len(points)-1][1],4),lll[layer],nice_path,0,1))
            else:
                for i in range(len(points)-1):
                    lines.append((round(points[i][0],4),  round(points[i][1],4), round(points[i+1][0],4), round(points[i+1][1],4),0,nice_path,0,1))
                lines.append((round(points[0][0],4),  round(points[0][1],4), round(points[len(points)-1][0],4), round(points[len(points)-1][1],4),0,nice_path,0,1))
           
    return lines

def distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
def find_closest_pointt(lines, target_point,nums):### возвращает точку, индекс линии, старт (1) или конец(0), и растояние до нее  
    closest_point = None
    min_distance = float('inf')
    I = 1
    
    mode = 1
    for i,line in enumerate(lines):
        if i in nums:
            m = 1
            for point in [line[0], line[1]]:
                dist = distance(point, target_point)
                if dist < min_distance:
                    min_distance = dist
                    closest_point = point
                    mode = m
                    I = i
                m = 0
            

    return closest_point,I,mode,min_distance
def find_closest_lines(lines, target_point,nums):
    
    
    Nums = set(nums)
    lins = []

    closest_point,I,mode,min_distance = find_closest_pointt(lines, target_point,Nums)

    current_point = closest_point

    while 1:
        closest_point,I,mode,min_distance = find_closest_pointt(lines, current_point,Nums)

        if min_distance < 1:
            Nums.remove(I)
            lins.append(I)
            if mode:
                current_point = lines[I][1]
            else:
                current_point = lines[I][0]
        else:
            return lins

def dxf_to_svg(dxf_file, svg_file): 

    doc = ezdxf.readfile(dxf_file)
    dwg = svgwrite.Drawing(svg_file, profile='tiny')

    for entity in doc.modelspace().query('LINE'):
        start = entity.dxf.start
        end = entity.dxf.end
        dwg.add(dwg.line(start=(start.x, start.y), end=(end.x, end.y), stroke=svgwrite.rgb(0, 0, 0, '%')))
    dwg.save()
def save_as_gcode():
    dpg.show_item("file_dialog_id2")
def callback_to_gcode(sender, app_data, user_data):
    
    current_file = app_data['file_path_name']
    gcode_lines = []

    gcode_lines.append("G90")
    gcode_lines.append("M4 S0")
    rec = db.get_records('lines')

    ts = [row[5] for row in rec]
    lines = [[(row[1],row[2]),(row[3],row[4])] for row in rec]
    set0 = {index for index, value in enumerate(ts) if value == 0}
    set1 = {index for index, value in enumerate(ts) if value == 1}
    set2 = {index for index, value in enumerate(ts) if value == 2}
    set3 = {index for index, value in enumerate(ts) if value == 3}
    set4 = {index for index, value in enumerate(ts) if value == 4}
    sets = []
    sets.append(set0)
    sets.append(set1)
    sets.append(set2)
    sets.append(set3)
    sets.append(set4)
    
    current_start = (0,0)
    h = 1
    for sett in sets:

        power = dpg.get_value(f"{h}1_value")
        speed = dpg.get_value(f"{h}_value")
        while sett:

            p,j,m,d = find_closest_pointt(lines,current_start,sett)
            if abs(current_start[0] - p[0]) > 0.01 or abs(current_start[1] - p[1]) > 0.01:
                gcode_lines.append(f"S0")
                gcode_lines.append(f"G0 X{round(p[0],4)} Y{round(p[1],4)}")         
            if m:
                gcode_lines.append(f"S{speed}")
                gcode_lines.append(f"G1 X{round(lines[j][1][0],4)} Y{round(lines[j][1][1],4)}F{power}")
                gcode_lines.append(f"S0")

                current_start = lines[j][1]
            else:
                gcode_lines.append(f"S{speed}")
                gcode_lines.append(f"G1 X{round(lines[j][0][0],4)} Y{round(lines[j][0][1],4)}F{power}")
                gcode_lines.append(f"S0")
                current_start = lines[j][0]

            sett.remove(j)
        h+=1
    

    gcode_lines.append(f"M5 S0")
    with open(current_file, 'w') as f:
        f.write("\n".join(gcode_lines))

    dpg.set_value('multiline_input',"\n".join(gcode_lines))
def save_as_dxf():
    dpg.show_item("file_dialog_id1")
def save_dxf(sender, app_data, user_data):
    current_file = app_data['file_path_name']
    doc = ezdxf.new()
    msp = doc.modelspace()
    rec = db.get_records('lines')

    ts = [row[5] for row in rec]
    lines = [[(row[1],row[2]),(row[3],row[4])] for row in rec]
    set0 = {index for index, value in enumerate(ts) if value == 0}
    set1 = {index for index, value in enumerate(ts) if value == 1}
    set2 = {index for index, value in enumerate(ts) if value == 2}
    set3 = {index for index, value in enumerate(ts) if value == 3}
    set4 = {index for index, value in enumerate(ts) if value == 4}
    sets = []
    sets.append(set0)
    sets.append(set1)
    sets.append(set2)
    sets.append(set3)
    sets.append(set4)
    h = 0
    current_start = (0,0)
    for sett in sets:
        power = dpg.get_value(f"{h+1}1_value")
        speed = dpg.get_value(f"{h+1}_value")
        
        layer_name = f"power{power}speed{speed}"
        doc.layers.new(name=layer_name, dxfattribs={'color': 7})  # 7 - цвет белый
        while sett:
            p,j,m,d = find_closest_pointt(lines,current_start,sett)

            if m:
                msp.add_line(lines[j][0], lines[j][1], dxfattribs={'layer': layer_name})
                current_start = lines[j][1]
            else:
                msp.add_line(lines[j][1], lines[j][0], dxfattribs={'layer': layer_name})
                current_start = lines[j][0]

            sett.remove(j)
        h+=1
    doc.saveas(current_file)


def check_callback(sender):
    for i in ['color_1','color_2','color_3','color_4','color_5']:
         if i != sender:
              dpg.set_value(i,False)

def print_me(sender):
    print(f"Menu Item: {sender}")

digit_lines = {
        '0': [(( 1.8 ,  6.6  ), ( 2.7 , 6.6 )),
(( 1.4 ,  6.5  ), ( 3.0 , 6.5 )),
(( 1.2 ,  6.4  ), ( 3.2 , 6.4 )),
(( 1.1 ,  6.3  ), ( 1.7 , 6.3 )),
(( 2.4 ,  6.3  ), ( 3.3 , 6.3 )),
(( 0.9 ,  6.2  ), ( 1.5 , 6.2 )),
(( 2.6 ,  6.2  ), ( 3.5 , 6.2 )),
(( 0.8 ,  6.1  ), ( 1.4 , 6.1 )),
(( 2.7 ,  6.1  ), ( 3.6 , 6.1 )),
(( 0.7 ,  6.0  ), ( 1.3 , 6.0 )),
(( 2.8 ,  6.0  ), ( 3.7 , 6.0 )),
(( 0.7 ,  5.9  ), ( 1.2 , 5.9 )),
(( 2.9 ,  5.9  ), ( 3.7 , 5.9 )),
(( 0.6 ,  5.8  ), ( 1.2 , 5.8 )),
(( 3.0 ,  5.8  ), ( 3.8 , 5.8 )),
(( 0.5 ,  5.7  ), ( 1.1 , 5.7 )),
(( 3.0 ,  5.7  ), ( 3.8 , 5.7 )),
(( 0.5 ,  5.6  ), ( 1.1 , 5.6 )),
(( 3.1 ,  5.6  ), ( 3.9 , 5.6 )),
(( 0.4 ,  5.5  ), ( 1.1 , 5.5 )),
(( 3.1 ,  5.5  ), ( 3.9 , 5.5 )),
(( 0.4 ,  5.4  ), ( 1.0 , 5.4 )),
(( 3.1 ,  5.4  ), ( 4.0 , 5.4 )),
(( 0.3 ,  5.3  ), ( 1.0 , 5.3 )),
(( 3.2 ,  5.3  ), ( 4.0 , 5.3 )),
(( 0.3 ,  5.2  ), ( 1.0 , 5.2 )),
(( 3.2 ,  5.2  ), ( 4.0 , 5.2 )),
(( 0.2 ,  5.1  ), ( 1.0 , 5.1 )),
(( 3.2 ,  5.1  ), ( 4.1 , 5.1 )),
(( 0.2 ,  5.0  ), ( 0.9 , 5.0 )),
(( 3.2 ,  5.0  ), ( 4.1 , 5.0 )),
(( 0.2 ,  4.9  ), ( 0.9 , 4.9 )),
(( 3.3 ,  4.9  ), ( 4.1 , 4.9 )),
(( 0.2 ,  4.8  ), ( 0.9 , 4.8 )),
(( 3.3 ,  4.8  ), ( 4.1 , 4.8 )),
(( 0.1 ,  4.7  ), ( 0.9 , 4.7 )),
(( 3.3 ,  4.7  ), ( 4.1 , 4.7 )),
(( 0.1 ,  4.6  ), ( 0.9 , 4.6 )),
(( 3.3 ,  4.6  ), ( 4.2 , 4.6 )),
(( 0.1 ,  4.5  ), ( 0.9 , 4.5 )),
(( 3.3 ,  4.5  ), ( 4.2 , 4.5 )),
(( 0.1 ,  4.4  ), ( 0.9 , 4.4 )),
(( 3.3 ,  4.4  ), ( 4.2 , 4.4 )),
(( 0.1 ,  4.3  ), ( 0.9 , 4.3 )),
(( 3.3 ,  4.3  ), ( 4.2 , 4.3 )),
(( 0.1 ,  4.2  ), ( 0.9 , 4.2 )),
(( 3.3 ,  4.2  ), ( 4.2 , 4.2 )),
(( 0.1 ,  4.1  ), ( 0.9 , 4.1 )),
(( 3.4 ,  4.1  ), ( 4.2 , 4.1 )),
(( 0.0 ,  4.0  ), ( 0.8 , 4.0 )),
(( 3.4 ,  4.0  ), ( 4.2 , 4.0 )),
(( 0.0 ,  3.9  ), ( 0.8 , 3.9 )),
(( 3.4 ,  3.9  ), ( 4.2 , 3.9 )),
(( 0.0 ,  3.8  ), ( 0.8 , 3.8 )),
(( 3.4 ,  3.8  ), ( 4.2 , 3.8 )),
(( 0.0 ,  3.7  ), ( 0.8 , 3.7 )),
(( 3.4 ,  3.7  ), ( 4.2 , 3.7 )),
(( 0.0 ,  3.6  ), ( 0.8 , 3.6 )),
(( 3.4 ,  3.6  ), ( 4.2 , 3.6 )),
(( 0.0 ,  3.5  ), ( 0.8 , 3.5 )),
(( 3.4 ,  3.5  ), ( 4.2 , 3.5 )),
(( 0.0 ,  3.4  ), ( 0.8 , 3.4 )),
(( 3.4 ,  3.4  ), ( 4.2 , 3.4 )),
(( 0.0 ,  3.3  ), ( 0.8 , 3.3 )),
(( 3.4 ,  3.3  ), ( 4.2 , 3.3 )),
(( 0.0 ,  3.2  ), ( 0.8 , 3.2 )),
(( 3.4 ,  3.2  ), ( 4.2 , 3.2 )),
(( 0.0 ,  3.1  ), ( 0.8 , 3.1 )),
(( 3.4 ,  3.1  ), ( 4.2 , 3.1 )),
(( 0.0 ,  3.0  ), ( 0.8 , 3.0 )),
(( 3.4 ,  3.0  ), ( 4.2 , 3.0 )),
(( 0.0 ,  2.9  ), ( 0.8 , 2.9 )),
(( 3.4 ,  2.9  ), ( 4.2 , 2.9 )),
(( 0.0 ,  2.8  ), ( 0.9 , 2.8 )),
(( 3.4 ,  2.8  ), ( 4.2 , 2.8 )),
(( 0.0 ,  2.7  ), ( 0.9 , 2.7 )),
(( 3.4 ,  2.7  ), ( 4.2 , 2.7 )),
(( 0.0 ,  2.6  ), ( 0.9 , 2.6 )),
(( 3.4 ,  2.6  ), ( 4.2 , 2.6 )),
(( 0.0 ,  2.5  ), ( 0.9 , 2.5 )),
(( 3.4 ,  2.5  ), ( 4.1 , 2.5 )),
(( 0.0 ,  2.4  ), ( 0.9 , 2.4 )),
(( 3.4 ,  2.4  ), ( 4.1 , 2.4 )),
(( 0.1 ,  2.3  ), ( 0.9 , 2.3 )),
(( 3.4 ,  2.3  ), ( 4.1 , 2.3 )),
(( 0.1 ,  2.2  ), ( 0.9 , 2.2 )),
(( 3.3 ,  2.2  ), ( 4.1 , 2.2 )),
(( 0.1 ,  2.1  ), ( 0.9 , 2.1 )),
(( 3.3 ,  2.1  ), ( 4.1 , 2.1 )),
(( 0.1 ,  2.0  ), ( 0.9 , 2.0 )),
(( 3.3 ,  2.0  ), ( 4.1 , 2.0 )),
(( 0.1 ,  1.9  ), ( 0.9 , 1.9 )),
(( 3.3 ,  1.9  ), ( 4.0 , 1.9 )),
(( 0.1 ,  1.8  ), ( 0.9 , 1.8 )),
(( 3.3 ,  1.8  ), ( 4.0 , 1.8 )),
(( 0.1 ,  1.7  ), ( 1.0 , 1.7 )),
(( 3.3 ,  1.7  ), ( 4.0 , 1.7 )),
(( 0.1 ,  1.6  ), ( 1.0 , 1.6 )),
(( 3.3 ,  1.6  ), ( 4.0 , 1.6 )),
(( 0.2 ,  1.5  ), ( 1.0 , 1.5 )),
(( 3.3 ,  1.5  ), ( 3.9 , 1.5 )),
(( 0.2 ,  1.4  ), ( 1.0 , 1.4 )),
(( 3.2 ,  1.4  ), ( 3.9 , 1.4 )),
(( 0.2 ,  1.3  ), ( 1.0 , 1.3 )),
(( 3.2 ,  1.3  ), ( 3.9 , 1.3 )),
(( 0.3 ,  1.2  ), ( 1.1 , 1.2 )),
(( 3.2 ,  1.2  ), ( 3.8 , 1.2 )),
(( 0.3 ,  1.1  ), ( 1.1 , 1.1 )),
(( 3.2 ,  1.1  ), ( 3.8 , 1.1 )),
(( 0.4 ,  1.0  ), ( 1.2 , 1.0 )),
(( 3.1 ,  1.0  ), ( 3.7 , 1.0 )),
(( 0.4 ,  0.9  ), ( 1.2 , 0.9 )),
(( 3.1 ,  0.9  ), ( 3.7 , 0.9 )),
(( 0.5 ,  0.8  ), ( 1.2 , 0.8 )),
(( 3.0 ,  0.8  ), ( 3.6 , 0.8 )),
(( 0.5 ,  0.7  ), ( 1.3 , 0.7 )),
(( 3.0 ,  0.7  ), ( 3.5 , 0.7 )),
(( 0.6 ,  0.6  ), ( 1.4 , 0.6 )),
(( 2.9 ,  0.6  ), ( 3.5 , 0.6 )),
(( 0.7 ,  0.5  ), ( 1.5 , 0.5 )),
(( 2.8 ,  0.5  ), ( 3.4 , 0.5 )),
(( 0.8 ,  0.4  ), ( 1.6 , 0.4 )),
(( 2.7 ,  0.4  ), ( 3.3 , 0.4 )),
(( 0.9 ,  0.3  ), ( 1.8 , 0.3 )),
(( 2.5 ,  0.3  ), ( 3.1 , 0.3 )),
(( 1.1 ,  0.2  ), ( 3.0 , 0.2 )),
(( 1.2 ,  0.1  ), ( 2.8 , 0.1 )),
(( 1.6 ,  0.0  ), ( 2.4 , 0.0 ))],
        '1': [(( 2.4 ,  6.5  ), ( 2.6 , 6.5 )),
(( 2.2 ,  6.4  ), ( 2.7 , 6.4 )),
(( 2.0 ,  6.3  ), ( 2.7 , 6.3 )),
(( 1.8 ,  6.2  ), ( 2.7 , 6.2 )),
(( 1.5 ,  6.1  ), ( 2.7 , 6.1 )),
(( 1.3 ,  6.0  ), ( 2.6 , 6.0 )),
(( 1.0 ,  5.9  ), ( 2.6 , 5.9 )),
(( 0.7 ,  5.8  ), ( 2.6 , 5.8 )),
(( 0.4 ,  5.7  ), ( 1.5 , 5.7 )),
(( 1.7 ,  5.7  ), ( 2.6 , 5.7 )),
(( 0.1 ,  5.6  ), ( 1.3 , 5.6 )),
(( 1.8 ,  5.6  ), ( 2.6 , 5.6 )),
(( 0.0 ,  5.5  ), ( 1.0 , 5.5 )),
(( 1.8 ,  5.5  ), ( 2.6 , 5.5 )),
(( 0.0 ,  5.4  ), ( 0.8 , 5.4 )),
(( 1.8 ,  5.4  ), ( 2.6 , 5.4 )),
(( 0.0 ,  5.3  ), ( 0.6 , 5.3 )),
(( 1.8 ,  5.3  ), ( 2.6 , 5.3 )),
(( 0.0 ,  5.2  ), ( 0.4 , 5.2 )),
(( 1.8 ,  5.2  ), ( 2.6 , 5.2 )),
(( 0.1 ,  5.1  ), ( 0.2 , 5.1 )),
(( 1.8 ,  5.1  ), ( 2.6 , 5.1 )),
(( 1.8 ,  5.0  ), ( 2.6 , 5.0 )),
(( 1.8 ,  4.9  ), ( 2.6 , 4.9 )),
(( 1.8 ,  4.8  ), ( 2.6 , 4.8 )),
(( 1.8 ,  4.7  ), ( 2.6 , 4.7 )),
(( 1.8 ,  4.6  ), ( 2.6 , 4.6 )),
(( 1.8 ,  4.5  ), ( 2.6 , 4.5 )),
(( 1.8 ,  4.4  ), ( 2.6 , 4.4 )),
(( 1.8 ,  4.3  ), ( 2.6 , 4.3 )),
(( 1.8 ,  4.2  ), ( 2.6 , 4.2 )),
(( 1.8 ,  4.1  ), ( 2.6 , 4.1 )),
(( 1.8 ,  4.0  ), ( 2.6 , 4.0 )),
(( 1.8 ,  3.9  ), ( 2.6 , 3.9 )),
(( 1.8 ,  3.8  ), ( 2.6 , 3.8 )),
(( 1.8 ,  3.7  ), ( 2.6 , 3.7 )),
(( 1.8 ,  3.6  ), ( 2.6 , 3.6 )),
(( 1.8 ,  3.5  ), ( 2.6 , 3.5 )),
(( 1.8 ,  3.4  ), ( 2.6 , 3.4 )),
(( 1.8 ,  3.3  ), ( 2.6 , 3.3 )),
(( 1.8 ,  3.2  ), ( 2.6 , 3.2 )),
(( 1.8 ,  3.1  ), ( 2.6 , 3.1 )),
(( 1.8 ,  3.0  ), ( 2.6 , 3.0 )),
(( 1.8 ,  2.9  ), ( 2.6 , 2.9 )),
(( 1.8 ,  2.8  ), ( 2.6 , 2.8 )),
(( 1.8 ,  2.7  ), ( 2.6 , 2.7 )),
(( 1.8 ,  2.6  ), ( 2.6 , 2.6 )),
(( 1.8 ,  2.5  ), ( 2.6 , 2.5 )),
(( 1.8 ,  2.4  ), ( 2.6 , 2.4 )),
(( 1.8 ,  2.3  ), ( 2.6 , 2.3 )),
(( 1.8 ,  2.2  ), ( 2.6 , 2.2 )),
(( 1.8 ,  2.1  ), ( 2.6 , 2.1 )),
(( 1.7 ,  2.0  ), ( 2.6 , 2.0 )),
(( 1.7 ,  1.9  ), ( 2.6 , 1.9 )),
(( 1.7 ,  1.8  ), ( 2.6 , 1.8 )),
(( 1.7 ,  1.7  ), ( 2.6 , 1.7 )),
(( 1.7 ,  1.6  ), ( 2.6 , 1.6 )),
(( 1.7 ,  1.5  ), ( 2.6 , 1.5 )),
(( 1.7 ,  1.4  ), ( 2.6 , 1.4 )),
(( 1.7 ,  1.3  ), ( 2.6 , 1.3 )),
(( 1.7 ,  1.2  ), ( 2.6 , 1.2 )),
(( 1.7 ,  1.1  ), ( 2.6 , 1.1 )),
(( 1.7 ,  1.0  ), ( 2.6 , 1.0 )),
(( 1.7 ,  0.9  ), ( 2.6 , 0.9 )),
(( 1.7 ,  0.8  ), ( 2.6 , 0.8 )),
(( 1.7 ,  0.7  ), ( 2.6 , 0.7 )),
(( 1.7 ,  0.6  ), ( 2.6 , 0.6 )),
(( 1.7 ,  0.5  ), ( 2.7 , 0.5 )),
(( 1.6 ,  0.4  ), ( 2.7 , 0.4 )),
(( 1.3 ,  0.3  ), ( 2.9 , 0.3 )),
(( 0.5 ,  0.2  ), ( 3.8 , 0.2 )),
(( 0.5 ,  0.1  ), ( 3.8 , 0.1 )),
(( 0.5 ,  0.0  ), ( 3.8 , 0.0 ))],
        '2': [(( 1.6 ,  6.5  ), ( 2.6 , 6.5 )),
(( 1.2 ,  6.4  ), ( 2.9 , 6.4 )),
(( 1.0 ,  6.3  ), ( 3.2 , 6.3 )),
(( 0.8 ,  6.2  ), ( 3.3 , 6.2 )),
(( 0.7 ,  6.1  ), ( 3.5 , 6.1 )),
(( 0.6 ,  6.0  ), ( 3.6 , 6.0 )),
(( 0.5 ,  5.9  ), ( 3.7 , 5.9 )),
(( 0.4 ,  5.8  ), ( 3.7 , 5.8 )),
(( 0.4 ,  5.7  ), ( 1.4 , 5.7 )),
(( 2.3 ,  5.7  ), ( 3.8 , 5.7 )),
(( 0.4 ,  5.6  ), ( 1.1 , 5.6 )),
(( 2.5 ,  5.6  ), ( 3.9 , 5.6 )),
(( 0.4 ,  5.5  ), ( 1.0 , 5.5 )),
(( 2.7 ,  5.5  ), ( 3.9 , 5.5 )),
(( 0.4 ,  5.4  ), ( 0.8 , 5.4 )),
(( 2.8 ,  5.4  ), ( 4.0 , 5.4 )),
(( 0.3 ,  5.3  ), ( 0.8 , 5.3 )),
(( 2.9 ,  5.3  ), ( 4.0 , 5.3 )),
(( 0.3 ,  5.2  ), ( 0.7 , 5.2 )),
(( 3.0 ,  5.2  ), ( 4.0 , 5.2 )),
(( 0.3 ,  5.1  ), ( 0.7 , 5.1 )),
(( 3.0 ,  5.1  ), ( 4.0 , 5.1 )),
(( 0.3 ,  5.0  ), ( 0.7 , 5.0 )),
(( 3.1 ,  5.0  ), ( 4.0 , 5.0 )),
(( 0.3 ,  4.9  ), ( 0.6 , 4.9 )),
(( 3.1 ,  4.9  ), ( 4.1 , 4.9 )),
(( 0.3 ,  4.8  ), ( 0.6 , 4.8 )),
(( 3.1 ,  4.8  ), ( 4.1 , 4.8 )),
(( 0.3 ,  4.7  ), ( 0.6 , 4.7 )),
(( 3.1 ,  4.7  ), ( 4.1 , 4.7 )),
(( 0.3 ,  4.6  ), ( 0.6 , 4.6 )),
(( 3.2 ,  4.6  ), ( 4.1 , 4.6 )),
(( 0.2 ,  4.5  ), ( 0.5 , 4.5 )),
(( 3.2 ,  4.5  ), ( 4.0 , 4.5 )),
(( 3.2 ,  4.4  ), ( 4.0 , 4.4 )),
(( 3.2 ,  4.3  ), ( 4.0 , 4.3 )),
(( 3.2 ,  4.2  ), ( 4.0 , 4.2 )),
(( 3.1 ,  4.1  ), ( 4.0 , 4.1 )),
(( 3.1 ,  4.0  ), ( 3.9 , 4.0 )),
(( 3.1 ,  3.9  ), ( 3.9 , 3.9 )),
(( 3.1 ,  3.8  ), ( 3.8 , 3.8 )),
(( 3.0 ,  3.7  ), ( 3.8 , 3.7 )),
(( 3.0 ,  3.6  ), ( 3.7 , 3.6 )),
(( 2.9 ,  3.5  ), ( 3.7 , 3.5 )),
(( 2.9 ,  3.4  ), ( 3.6 , 3.4 )),
(( 2.8 ,  3.3  ), ( 3.5 , 3.3 )),
(( 2.8 ,  3.2  ), ( 3.5 , 3.2 )),
(( 2.7 ,  3.1  ), ( 3.4 , 3.1 )),
(( 2.6 ,  3.0  ), ( 3.3 , 3.0 )),
(( 2.6 ,  2.9  ), ( 3.2 , 2.9 )),
(( 2.5 ,  2.8  ), ( 3.1 , 2.8 )),
(( 2.4 ,  2.7  ), ( 3.0 , 2.7 )),
(( 2.3 ,  2.6  ), ( 2.9 , 2.6 )),
(( 2.2 ,  2.5  ), ( 2.8 , 2.5 )),
(( 2.1 ,  2.4  ), ( 2.7 , 2.4 )),
(( 2.1 ,  2.3  ), ( 2.6 , 2.3 )),
(( 2.0 ,  2.2  ), ( 2.5 , 2.2 )),
(( 1.9 ,  2.1  ), ( 2.4 , 2.1 )),
(( 1.8 ,  2.0  ), ( 2.3 , 2.0 )),
(( 1.7 ,  1.9  ), ( 2.2 , 1.9 )),
(( 1.6 ,  1.8  ), ( 2.1 , 1.8 )),
(( 1.5 ,  1.7  ), ( 2.0 , 1.7 )),
(( 1.4 ,  1.6  ), ( 1.9 , 1.6 )),
(( 1.3 ,  1.5  ), ( 1.8 , 1.5 )),
(( 1.2 ,  1.4  ), ( 1.7 , 1.4 )),
(( 1.1 ,  1.3  ), ( 1.6 , 1.3 )),
(( 1.0 ,  1.2  ), ( 1.5 , 1.2 )),
(( 0.9 ,  1.1  ), ( 1.4 , 1.1 )),
(( 0.8 ,  1.0  ), ( 1.3 , 1.0 )),
(( 0.7 ,  0.9  ), ( 1.2 , 0.9 )),
(( 0.6 ,  0.8  ), ( 1.1 , 0.8 )),
(( 4.0 ,  0.8  ), ( 4.4 , 0.8 )),
(( 0.5 ,  0.7  ), ( 4.4 , 0.7 )),
(( 0.4 ,  0.6  ), ( 4.4 , 0.6 )),
(( 0.3 ,  0.5  ), ( 4.4 , 0.5 )),
(( 0.2 ,  0.4  ), ( 4.4 , 0.4 )),
(( 0.1 ,  0.3  ), ( 4.4 , 0.3 )),
(( 0.0 ,  0.2  ), ( 4.4 , 0.2 )),
(( 0.0 ,  0.1  ), ( 4.4 , 0.1 )),
(( 0.0 ,  0.0  ), ( 4.4 , 0.0 ))],
        '3': [(( 1.5 ,  6.6  ), ( 2.5 , 6.6 )),
(( 1.2 ,  6.5  ), ( 2.8 , 6.5 )),
(( 1.0 ,  6.4  ), ( 3.0 , 6.4 )),
(( 0.8 ,  6.3  ), ( 3.2 , 6.3 )),
(( 0.7 ,  6.2  ), ( 3.3 , 6.2 )),
(( 0.6 ,  6.1  ), ( 3.4 , 6.1 )),
(( 0.5 ,  6.0  ), ( 3.5 , 6.0 )),
(( 0.4 ,  5.9  ), ( 1.4 , 5.9 )),
(( 2.2 ,  5.9  ), ( 3.5 , 5.9 )),
(( 0.3 ,  5.8  ), ( 1.1 , 5.8 )),
(( 2.4 ,  5.8  ), ( 3.6 , 5.8 )),
(( 0.3 ,  5.7  ), ( 1.0 , 5.7 )),
(( 2.5 ,  5.7  ), ( 3.6 , 5.7 )),
(( 0.3 ,  5.6  ), ( 0.9 , 5.6 )),
(( 2.6 ,  5.6  ), ( 3.7 , 5.6 )),
(( 0.3 ,  5.5  ), ( 0.8 , 5.5 )),
(( 2.7 ,  5.5  ), ( 3.7 , 5.5 )),
(( 0.3 ,  5.4  ), ( 0.7 , 5.4 )),
(( 2.8 ,  5.4  ), ( 3.7 , 5.4 )),
(( 0.2 ,  5.3  ), ( 0.6 , 5.3 )),
(( 2.8 ,  5.3  ), ( 3.7 , 5.3 )),
(( 0.2 ,  5.2  ), ( 0.6 , 5.2 )),
(( 2.9 ,  5.2  ), ( 3.7 , 5.2 )),
(( 0.2 ,  5.1  ), ( 0.5 , 5.1 )),
(( 2.9 ,  5.1  ), ( 3.7 , 5.1 )),
(( 0.2 ,  5.0  ), ( 0.5 , 5.0 )),
(( 2.9 ,  5.0  ), ( 3.7 , 5.0 )),
(( 0.2 ,  4.9  ), ( 0.5 , 4.9 )),
(( 2.9 ,  4.9  ), ( 3.7 , 4.9 )),
(( 0.2 ,  4.8  ), ( 0.5 , 4.8 )),
(( 2.9 ,  4.8  ), ( 3.6 , 4.8 )),
(( 0.1 ,  4.7  ), ( 0.4 , 4.7 )),
(( 2.9 ,  4.7  ), ( 3.6 , 4.7 )),
(( 0.1 ,  4.6  ), ( 0.4 , 4.6 )),
(( 2.9 ,  4.6  ), ( 3.6 , 4.6 )),
(( 2.9 ,  4.5  ), ( 3.5 , 4.5 )),
(( 2.9 ,  4.4  ), ( 3.4 , 4.4 )),
(( 2.8 ,  4.3  ), ( 3.4 , 4.3 )),
(( 2.8 ,  4.2  ), ( 3.3 , 4.2 )),
(( 2.7 ,  4.1  ), ( 3.2 , 4.1 )),
(( 2.6 ,  4.0  ), ( 3.1 , 4.0 )),
(( 2.6 ,  3.9  ), ( 3.0 , 3.9 )),
(( 2.4 ,  3.8  ), ( 2.9 , 3.8 )),
(( 2.3 ,  3.7  ), ( 2.8 , 3.7 )),
(( 2.0 ,  3.6  ), ( 3.1 , 3.6 )),
(( 1.2 ,  3.5  ), ( 3.4 , 3.5 )),
(( 1.2 ,  3.4  ), ( 3.6 , 3.4 )),
(( 1.2 ,  3.3  ), ( 3.8 , 3.3 )),
(( 1.2 ,  3.2  ), ( 1.6 , 3.2 )),
(( 2.5 ,  3.2  ), ( 3.9 , 3.2 )),
(( 1.2 ,  3.1  ), ( 1.3 , 3.1 )),
(( 2.7 ,  3.1  ), ( 4.0 , 3.1 )),
(( 2.9 ,  3.0  ), ( 4.0 , 3.0 )),
(( 3.0 ,  2.9  ), ( 4.1 , 2.9 )),
(( 3.1 ,  2.8  ), ( 4.1 , 2.8 )),
(( 3.1 ,  2.7  ), ( 4.2 , 2.7 )),
(( 3.2 ,  2.6  ), ( 4.2 , 2.6 )),
(( 3.2 ,  2.5  ), ( 4.2 , 2.5 )),
(( 3.3 ,  2.4  ), ( 4.2 , 2.4 )),
(( 3.3 ,  2.3  ), ( 4.2 , 2.3 )),
(( 3.3 ,  2.2  ), ( 4.2 , 2.2 )),
(( 3.3 ,  2.1  ), ( 4.2 , 2.1 )),
(( 3.3 ,  2.0  ), ( 4.2 , 2.0 )),
(( 3.3 ,  1.9  ), ( 4.2 , 1.9 )),
(( 3.3 ,  1.8  ), ( 4.2 , 1.8 )),
(( 3.3 ,  1.7  ), ( 4.1 , 1.7 )),
(( 0.1 ,  1.6  ), ( 0.2 , 1.6 )),
(( 3.3 ,  1.6  ), ( 4.1 , 1.6 )),
(( 0.0 ,  1.5  ), ( 0.3 , 1.5 )),
(( 3.3 ,  1.5  ), ( 4.0 , 1.5 )),
(( 0.0 ,  1.4  ), ( 0.3 , 1.4 )),
(( 3.3 ,  1.4  ), ( 4.0 , 1.4 )),
(( 0.0 ,  1.3  ), ( 0.3 , 1.3 )),
(( 3.3 ,  1.3  ), ( 3.9 , 1.3 )),
(( 0.1 ,  1.2  ), ( 0.4 , 1.2 )),
(( 3.2 ,  1.2  ), ( 3.9 , 1.2 )),
(( 0.1 ,  1.1  ), ( 0.4 , 1.1 )),
(( 3.2 ,  1.1  ), ( 3.8 , 1.1 )),
(( 0.1 ,  1.0  ), ( 0.5 , 1.0 )),
(( 3.2 ,  1.0  ), ( 3.7 , 1.0 )),
(( 0.1 ,  0.9  ), ( 0.6 , 0.9 )),
(( 3.1 ,  0.9  ), ( 3.6 , 0.9 )),
(( 0.2 ,  0.8  ), ( 0.6 , 0.8 )),
(( 3.0 ,  0.8  ), ( 3.5 , 0.8 )),
(( 0.2 ,  0.7  ), ( 0.7 , 0.7 )),
(( 3.0 ,  0.7  ), ( 3.4 , 0.7 )),
(( 0.2 ,  0.6  ), ( 0.8 , 0.6 )),
(( 2.9 ,  0.6  ), ( 3.3 , 0.6 )),
(( 0.3 ,  0.5  ), ( 0.9 , 0.5 )),
(( 2.7 ,  0.5  ), ( 3.2 , 0.5 )),
(( 0.3 ,  0.4  ), ( 1.1 , 0.4 )),
(( 2.6 ,  0.4  ), ( 3.0 , 0.4 )),
(( 0.3 ,  0.3  ), ( 1.3 , 0.3 )),
(( 2.3 ,  0.3  ), ( 2.9 , 0.3 )),
(( 0.3 ,  0.2  ), ( 2.7 , 0.2 )),
(( 0.6 ,  0.1  ), ( 2.4 , 0.1 )),
(( 1.0 ,  0.0  ), ( 2.0 , 0.0 ))],
        '4': [(( 3.3 ,  6.5  ), ( 3.5 , 6.5 )),
(( 3.0 ,  6.4  ), ( 3.5 , 6.4 )),
(( 2.8 ,  6.3  ), ( 3.5 , 6.3 )),
(( 2.7 ,  6.2  ), ( 3.5 , 6.2 )),
(( 2.6 ,  6.1  ), ( 3.5 , 6.1 )),
(( 2.5 ,  6.0  ), ( 3.5 , 6.0 )),
(( 2.5 ,  5.9  ), ( 3.5 , 5.9 )),
(( 2.4 ,  5.8  ), ( 3.5 , 5.8 )),
(( 2.3 ,  5.7  ), ( 3.5 , 5.7 )),
(( 2.3 ,  5.6  ), ( 3.5 , 5.6 )),
(( 2.2 ,  5.5  ), ( 3.5 , 5.5 )),
(( 2.1 ,  5.4  ), ( 2.6 , 5.4 )),
(( 2.7 ,  5.4  ), ( 3.5 , 5.4 )),
(( 2.1 ,  5.3  ), ( 2.5 , 5.3 )),
(( 2.7 ,  5.3  ), ( 3.5 , 5.3 )),
(( 2.0 ,  5.2  ), ( 2.4 , 5.2 )),
(( 2.7 ,  5.2  ), ( 3.5 , 5.2 )),
(( 1.9 ,  5.1  ), ( 2.3 , 5.1 )),
(( 2.7 ,  5.1  ), ( 3.5 , 5.1 )),
(( 1.9 ,  5.0  ), ( 2.3 , 5.0 )),
(( 2.7 ,  5.0  ), ( 3.5 , 5.0 )),
(( 1.8 ,  4.9  ), ( 2.2 , 4.9 )),
(( 2.7 ,  4.9  ), ( 3.5 , 4.9 )),
(( 1.8 ,  4.8  ), ( 2.2 , 4.8 )),
(( 2.7 ,  4.8  ), ( 3.5 , 4.8 )),
(( 1.7 ,  4.7  ), ( 2.1 , 4.7 )),
(( 2.7 ,  4.7  ), ( 3.5 , 4.7 )),
(( 1.6 ,  4.6  ), ( 2.0 , 4.6 )),
(( 2.7 ,  4.6  ), ( 3.5 , 4.6 )),
(( 1.6 ,  4.5  ), ( 2.0 , 4.5 )),
(( 2.7 ,  4.5  ), ( 3.5 , 4.5 )),
(( 1.5 ,  4.4  ), ( 1.9 , 4.4 )),
(( 2.7 ,  4.4  ), ( 3.5 , 4.4 )),
(( 1.4 ,  4.3  ), ( 1.8 , 4.3 )),
(( 2.7 ,  4.3  ), ( 3.5 , 4.3 )),
(( 1.4 ,  4.2  ), ( 1.8 , 4.2 )),
(( 2.7 ,  4.2  ), ( 3.5 , 4.2 )),
(( 1.3 ,  4.1  ), ( 1.7 , 4.1 )),
(( 2.7 ,  4.1  ), ( 3.5 , 4.1 )),
(( 1.3 ,  4.0  ), ( 1.7 , 4.0 )),
(( 2.7 ,  4.0  ), ( 3.5 , 4.0 )),
(( 1.2 ,  3.9  ), ( 1.6 , 3.9 )),
(( 2.7 ,  3.9  ), ( 3.5 , 3.9 )),
(( 1.1 ,  3.8  ), ( 1.5 , 3.8 )),
(( 2.7 ,  3.8  ), ( 3.5 , 3.8 )),
(( 1.1 ,  3.7  ), ( 1.5 , 3.7 )),
(( 2.7 ,  3.7  ), ( 3.5 , 3.7 )),
(( 1.0 ,  3.6  ), ( 1.4 , 3.6 )),
(( 2.7 ,  3.6  ), ( 3.5 , 3.6 )),
(( 1.0 ,  3.5  ), ( 1.4 , 3.5 )),
(( 2.7 ,  3.5  ), ( 3.5 , 3.5 )),
(( 0.9 ,  3.4  ), ( 1.3 , 3.4 )),
(( 2.7 ,  3.4  ), ( 3.5 , 3.4 )),
(( 0.8 ,  3.3  ), ( 1.2 , 3.3 )),
(( 2.7 ,  3.3  ), ( 3.5 , 3.3 )),
(( 0.8 ,  3.2  ), ( 1.2 , 3.2 )),
(( 2.7 ,  3.2  ), ( 3.5 , 3.2 )),
(( 0.7 ,  3.1  ), ( 1.1 , 3.1 )),
(( 2.7 ,  3.1  ), ( 3.5 , 3.1 )),
(( 0.7 ,  3.0  ), ( 1.1 , 3.0 )),
(( 2.7 ,  3.0  ), ( 3.5 , 3.0 )),
(( 0.6 ,  2.9  ), ( 1.0 , 2.9 )),
(( 2.7 ,  2.9  ), ( 3.5 , 2.9 )),
(( 0.5 ,  2.8  ), ( 1.0 , 2.8 )),
(( 2.7 ,  2.8  ), ( 3.5 , 2.8 )),
(( 0.5 ,  2.7  ), ( 0.9 , 2.7 )),
(( 2.7 ,  2.7  ), ( 3.5 , 2.7 )),
(( 0.4 ,  2.6  ), ( 0.8 , 2.6 )),
(( 2.7 ,  2.6  ), ( 3.5 , 2.6 )),
(( 0.4 ,  2.5  ), ( 0.8 , 2.5 )),
(( 2.7 ,  2.5  ), ( 3.5 , 2.5 )),
(( 0.3 ,  2.4  ), ( 0.7 , 2.4 )),
(( 2.7 ,  2.4  ), ( 3.5 , 2.4 )),
(( 0.3 ,  2.3  ), ( 0.7 , 2.3 )),
(( 2.7 ,  2.3  ), ( 3.5 , 2.3 )),
(( 0.2 ,  2.2  ), ( 4.5 , 2.2 )),
(( 0.1 ,  2.1  ), ( 4.5 , 2.1 )),
(( 0.1 ,  2.0  ), ( 4.5 , 2.0 )),
(( 0.0 ,  1.9  ), ( 4.5 , 1.9 )),
(( 0.1 ,  1.8  ), ( 4.5 , 1.8 )),
(( 0.1 ,  1.7  ), ( 4.5 , 1.7 )),
(( 2.7 ,  1.6  ), ( 3.5 , 1.6 )),
(( 2.7 ,  1.5  ), ( 3.5 , 1.5 )),
(( 2.7 ,  1.4  ), ( 3.5 , 1.4 )),
(( 2.7 ,  1.3  ), ( 3.5 , 1.3 )),
(( 2.7 ,  1.2  ), ( 3.5 , 1.2 )),
(( 2.7 ,  1.1  ), ( 3.5 , 1.1 )),
(( 2.6 ,  1.0  ), ( 3.5 , 1.0 )),
(( 2.6 ,  0.9  ), ( 3.5 , 0.9 )),
(( 2.6 ,  0.8  ), ( 3.5 , 0.8 )),
(( 2.6 ,  0.7  ), ( 3.5 , 0.7 )),
(( 2.6 ,  0.6  ), ( 3.5 , 0.6 )),
(( 2.6 ,  0.5  ), ( 3.5 , 0.5 )),
(( 2.5 ,  0.4  ), ( 3.6 , 0.4 )),
(( 2.2 ,  0.3  ), ( 3.8 , 0.3 )),
(( 1.5 ,  0.2  ), ( 4.5 , 0.2 )),
(( 1.5 ,  0.1  ), ( 4.5 , 0.1 )),
(( 1.5 ,  0.0  ), ( 4.5 , 0.0 ))],
        '5': [(( 0.6 ,  6.6  ), ( 1.1 , 6.6 )),
(( 3.5 ,  6.6  ), ( 4.0 , 6.6 )),
(( 0.6 ,  6.5  ), ( 4.0 , 6.5 )),
(( 0.6 ,  6.4  ), ( 4.0 , 6.4 )),
(( 0.6 ,  6.3  ), ( 4.0 , 6.3 )),
(( 0.6 ,  6.2  ), ( 4.0 , 6.2 )),
(( 0.7 ,  6.1  ), ( 4.0 , 6.1 )),
(( 0.7 ,  6.0  ), ( 4.0 , 6.0 )),
(( 0.7 ,  5.9  ), ( 4.0 , 5.9 )),
(( 0.7 ,  5.8  ), ( 1.1 , 5.8 )),
(( 0.7 ,  5.7  ), ( 1.1 , 5.7 )),
(( 0.7 ,  5.6  ), ( 1.1 , 5.6 )),
(( 0.7 ,  5.5  ), ( 1.1 , 5.5 )),
(( 0.7 ,  5.4  ), ( 1.0 , 5.4 )),
(( 0.7 ,  5.3  ), ( 1.0 , 5.3 )),
(( 0.7 ,  5.2  ), ( 1.0 , 5.2 )),
(( 0.7 ,  5.1  ), ( 1.0 , 5.1 )),
(( 0.7 ,  5.0  ), ( 1.0 , 5.0 )),
(( 0.7 ,  4.9  ), ( 1.0 , 4.9 )),
(( 0.7 ,  4.8  ), ( 1.0 , 4.8 )),
(( 0.7 ,  4.7  ), ( 1.0 , 4.7 )),
(( 0.7 ,  4.6  ), ( 1.0 , 4.6 )),
(( 0.7 ,  4.5  ), ( 1.0 , 4.5 )),
(( 0.7 ,  4.4  ), ( 1.0 , 4.4 )),
(( 1.9 ,  4.4  ), ( 2.9 , 4.4 )),
(( 0.7 ,  4.3  ), ( 1.0 , 4.3 )),
(( 1.6 ,  4.3  ), ( 3.2 , 4.3 )),
(( 0.7 ,  4.2  ), ( 1.0 , 4.2 )),
(( 1.4 ,  4.2  ), ( 3.4 , 4.2 )),
(( 0.7 ,  4.1  ), ( 1.0 , 4.1 )),
(( 1.2 ,  4.1  ), ( 3.6 , 4.1 )),
(( 0.7 ,  4.0  ), ( 1.0 , 4.0 )),
(( 1.1 ,  4.0  ), ( 3.7 , 4.0 )),
(( 0.7 ,  3.9  ), ( 3.8 , 3.9 )),
(( 0.6 ,  3.8  ), ( 3.9 , 3.8 )),
(( 0.6 ,  3.7  ), ( 1.6 , 3.7 )),
(( 2.5 ,  3.7  ), ( 4.0 , 3.7 )),
(( 0.6 ,  3.6  ), ( 1.3 , 3.6 )),
(( 2.8 ,  3.6  ), ( 4.1 , 3.6 )),
(( 0.6 ,  3.5  ), ( 1.2 , 3.5 )),
(( 2.9 ,  3.5  ), ( 4.1 , 3.5 )),
(( 0.6 ,  3.4  ), ( 1.0 , 3.4 )),
(( 3.0 ,  3.4  ), ( 4.2 , 3.4 )),
(( 0.6 ,  3.3  ), ( 0.9 , 3.3 )),
(( 3.2 ,  3.3  ), ( 4.2 , 3.3 )),
(( 0.7 ,  3.2  ), ( 0.8 , 3.2 )),
(( 3.2 ,  3.2  ), ( 4.2 , 3.2 )),
(( 3.3 ,  3.1  ), ( 4.3 , 3.1 )),
(( 3.4 ,  3.0  ), ( 4.3 , 3.0 )),
(( 3.4 ,  2.9  ), ( 4.3 , 2.9 )),
(( 3.4 ,  2.8  ), ( 4.3 , 2.8 )),
(( 3.5 ,  2.7  ), ( 4.3 , 2.7 )),
(( 3.5 ,  2.6  ), ( 4.3 , 2.6 )),
(( 3.5 ,  2.5  ), ( 4.3 , 2.5 )),
(( 3.5 ,  2.4  ), ( 4.3 , 2.4 )),
(( 3.5 ,  2.3  ), ( 4.3 , 2.3 )),
(( 3.5 ,  2.2  ), ( 4.3 , 2.2 )),
(( 3.5 ,  2.1  ), ( 4.3 , 2.1 )),
(( 3.5 ,  2.0  ), ( 4.3 , 2.0 )),
(( 3.5 ,  1.9  ), ( 4.2 , 1.9 )),
(( 3.5 ,  1.8  ), ( 4.2 , 1.8 )),
(( 3.5 ,  1.7  ), ( 4.2 , 1.7 )),
(( 0.1 ,  1.6  ), ( 0.3 , 1.6 )),
(( 3.4 ,  1.6  ), ( 4.1 , 1.6 )),
(( 0.0 ,  1.5  ), ( 0.4 , 1.5 )),
(( 3.4 ,  1.5  ), ( 4.1 , 1.5 )),
(( 0.1 ,  1.4  ), ( 0.4 , 1.4 )),
(( 3.4 ,  1.4  ), ( 4.0 , 1.4 )),
(( 0.1 ,  1.3  ), ( 0.5 , 1.3 )),
(( 3.3 ,  1.3  ), ( 4.0 , 1.3 )),
(( 0.2 ,  1.2  ), ( 0.6 , 1.2 )),
(( 3.3 ,  1.2  ), ( 3.9 , 1.2 )),
(( 0.2 ,  1.1  ), ( 0.6 , 1.1 )),
(( 3.2 ,  1.1  ), ( 3.8 , 1.1 )),
(( 0.2 ,  1.0  ), ( 0.7 , 1.0 )),
(( 3.1 ,  1.0  ), ( 3.7 , 1.0 )),
(( 0.3 ,  0.9  ), ( 0.8 , 0.9 )),
(( 3.0 ,  0.9  ), ( 3.7 , 0.9 )),
(( 0.3 ,  0.8  ), ( 0.9 , 0.8 )),
(( 2.8 ,  0.8  ), ( 3.6 , 0.8 )),
(( 0.4 ,  0.7  ), ( 1.0 , 0.7 )),
(( 2.7 ,  0.7  ), ( 3.5 , 0.7 )),
(( 0.4 ,  0.6  ), ( 1.3 , 0.6 )),
(( 2.4 ,  0.6  ), ( 3.3 , 0.6 )),
(( 0.4 ,  0.5  ), ( 3.2 , 0.5 )),
(( 0.5 ,  0.4  ), ( 3.1 , 0.4 )),
(( 0.5 ,  0.3  ), ( 2.9 , 0.3 )),
(( 0.6 ,  0.2  ), ( 2.7 , 0.2 )),
(( 0.8 ,  0.1  ), ( 2.4 , 0.1 )),
(( 1.1 ,  0.0  ), ( 2.0 , 0.0 ))],
        '6': [(( 3.3 ,  6.6  ), ( 3.5 , 6.6 )),
(( 3.0 ,  6.5  ), ( 3.7 , 6.5 )),
(( 2.7 ,  6.4  ), ( 3.6 , 6.4 )),
(( 2.5 ,  6.3  ), ( 3.3 , 6.3 )),
(( 2.3 ,  6.2  ), ( 3.1 , 6.2 )),
(( 2.1 ,  6.1  ), ( 2.9 , 6.1 )),
(( 1.9 ,  6.0  ), ( 2.7 , 6.0 )),
(( 1.8 ,  5.9  ), ( 2.5 , 5.9 )),
(( 1.7 ,  5.8  ), ( 2.4 , 5.8 )),
(( 1.5 ,  5.7  ), ( 2.3 , 5.7 )),
(( 1.4 ,  5.6  ), ( 2.1 , 5.6 )),
(( 1.3 ,  5.5  ), ( 2.0 , 5.5 )),
(( 1.2 ,  5.4  ), ( 1.9 , 5.4 )),
(( 1.1 ,  5.3  ), ( 1.8 , 5.3 )),
(( 1.0 ,  5.2  ), ( 1.7 , 5.2 )),
(( 0.9 ,  5.1  ), ( 1.7 , 5.1 )),
(( 0.8 ,  5.0  ), ( 1.6 , 5.0 )),
(( 0.8 ,  4.9  ), ( 1.5 , 4.9 )),
(( 0.7 ,  4.8  ), ( 1.4 , 4.8 )),
(( 0.6 ,  4.7  ), ( 1.4 , 4.7 )),
(( 0.6 ,  4.6  ), ( 1.3 , 4.6 )),
(( 0.5 ,  4.5  ), ( 1.3 , 4.5 )),
(( 0.5 ,  4.4  ), ( 1.2 , 4.4 )),
(( 0.4 ,  4.3  ), ( 1.2 , 4.3 )),
(( 0.4 ,  4.2  ), ( 1.2 , 4.2 )),
(( 0.3 ,  4.1  ), ( 1.1 , 4.1 )),
(( 0.3 ,  4.0  ), ( 1.1 , 4.0 )),
(( 2.1 ,  4.0  ), ( 3.0 , 4.0 )),
(( 0.3 ,  3.9  ), ( 1.1 , 3.9 )),
(( 1.8 ,  3.9  ), ( 3.3 , 3.9 )),
(( 0.2 ,  3.8  ), ( 1.1 , 3.8 )),
(( 1.7 ,  3.8  ), ( 3.5 , 3.8 )),
(( 0.2 ,  3.7  ), ( 1.0 , 3.7 )),
(( 1.5 ,  3.7  ), ( 3.6 , 3.7 )),
(( 0.2 ,  3.6  ), ( 1.0 , 3.6 )),
(( 1.4 ,  3.6  ), ( 3.7 , 3.6 )),
(( 0.1 ,  3.5  ), ( 1.0 , 3.5 )),
(( 1.3 ,  3.5  ), ( 3.8 , 3.5 )),
(( 0.1 ,  3.4  ), ( 1.0 , 3.4 )),
(( 1.1 ,  3.4  ), ( 1.7 , 3.4 )),
(( 2.5 ,  3.4  ), ( 3.9 , 3.4 )),
(( 0.1 ,  3.3  ), ( 1.5 , 3.3 )),
(( 2.7 ,  3.3  ), ( 4.0 , 3.3 )),
(( 0.1 ,  3.2  ), ( 1.3 , 3.2 )),
(( 2.9 ,  3.2  ), ( 4.0 , 3.2 )),
(( 0.1 ,  3.1  ), ( 1.2 , 3.1 )),
(( 3.0 ,  3.1  ), ( 4.1 , 3.1 )),
(( 0.1 ,  3.0  ), ( 1.1 , 3.0 )),
(( 3.1 ,  3.0  ), ( 4.1 , 3.0 )),
(( 0.0 ,  2.9  ), ( 1.1 , 2.9 )),
(( 3.2 ,  2.9  ), ( 4.1 , 2.9 )),
(( 0.0 ,  2.8  ), ( 1.0 , 2.8 )),
(( 3.2 ,  2.8  ), ( 4.1 , 2.8 )),
(( 0.0 ,  2.7  ), ( 1.0 , 2.7 )),
(( 3.2 ,  2.7  ), ( 4.2 , 2.7 )),
(( 0.0 ,  2.6  ), ( 0.9 , 2.6 )),
(( 3.3 ,  2.6  ), ( 4.2 , 2.6 )),
(( 0.0 ,  2.5  ), ( 0.9 , 2.5 )),
(( 3.3 ,  2.5  ), ( 4.2 , 2.5 )),
(( 0.0 ,  2.4  ), ( 0.9 , 2.4 )),
(( 3.3 ,  2.4  ), ( 4.2 , 2.4 )),
(( 0.0 ,  2.3  ), ( 0.9 , 2.3 )),
(( 3.4 ,  2.3  ), ( 4.2 , 2.3 )),
(( 0.0 ,  2.2  ), ( 0.9 , 2.2 )),
(( 3.4 ,  2.2  ), ( 4.2 , 2.2 )),
(( 0.0 ,  2.1  ), ( 0.9 , 2.1 )),
(( 3.4 ,  2.1  ), ( 4.2 , 2.1 )),
(( 0.1 ,  2.0  ), ( 0.9 , 2.0 )),
(( 3.4 ,  2.0  ), ( 4.2 , 2.0 )),
(( 0.1 ,  1.9  ), ( 0.9 , 1.9 )),
(( 3.4 ,  1.9  ), ( 4.2 , 1.9 )),
(( 0.1 ,  1.8  ), ( 0.9 , 1.8 )),
(( 3.4 ,  1.8  ), ( 4.2 , 1.8 )),
(( 0.1 ,  1.7  ), ( 0.9 , 1.7 )),
(( 3.4 ,  1.7  ), ( 4.1 , 1.7 )),
(( 0.1 ,  1.6  ), ( 0.9 , 1.6 )),
(( 3.4 ,  1.6  ), ( 4.1 , 1.6 )),
(( 0.1 ,  1.5  ), ( 1.0 , 1.5 )),
(( 3.3 ,  1.5  ), ( 4.1 , 1.5 )),
(( 0.2 ,  1.4  ), ( 1.0 , 1.4 )),
(( 3.3 ,  1.4  ), ( 4.0 , 1.4 )),
(( 0.2 ,  1.3  ), ( 1.0 , 1.3 )),
(( 3.3 ,  1.3  ), ( 4.0 , 1.3 )),
(( 0.2 ,  1.2  ), ( 1.0 , 1.2 )),
(( 3.3 ,  1.2  ), ( 4.0 , 1.2 )),
(( 0.3 ,  1.1  ), ( 1.1 , 1.1 )),
(( 3.3 ,  1.1  ), ( 3.9 , 1.1 )),
(( 0.3 ,  1.0  ), ( 1.1 , 1.0 )),
(( 3.2 ,  1.0  ), ( 3.9 , 1.0 )),
(( 0.4 ,  0.9  ), ( 1.2 , 0.9 )),
(( 3.2 ,  0.9  ), ( 3.8 , 0.9 )),
(( 0.4 ,  0.8  ), ( 1.2 , 0.8 )),
(( 3.1 ,  0.8  ), ( 3.7 , 0.8 )),
(( 0.5 ,  0.7  ), ( 1.3 , 0.7 )),
(( 3.0 ,  0.7  ), ( 3.6 , 0.7 )),
(( 0.6 ,  0.6  ), ( 1.4 , 0.6 )),
(( 3.0 ,  0.6  ), ( 3.5 , 0.6 )),
(( 0.7 ,  0.5  ), ( 1.5 , 0.5 )),
(( 2.9 ,  0.5  ), ( 3.4 , 0.5 )),
(( 0.8 ,  0.4  ), ( 1.6 , 0.4 )),
(( 2.7 ,  0.4  ), ( 3.3 , 0.4 )),
(( 0.9 ,  0.3  ), ( 1.8 , 0.3 )),
(( 2.5 ,  0.3  ), ( 3.2 , 0.3 )),
(( 1.1 ,  0.2  ), ( 3.0 , 0.2 )),
(( 1.3 ,  0.1  ), ( 2.8 , 0.1 )),
(( 1.6 ,  0.0  ), ( 2.5 , 0.0 ))],
        '7': [(( 0.1 ,  6.5  ), ( 0.5 , 6.5 )),
(( 3.8 ,  6.5  ), ( 4.4 , 6.5 )),
(( 0.0 ,  6.4  ), ( 4.4 , 6.4 )),
(( 0.0 ,  6.3  ), ( 4.4 , 6.3 )),
(( 0.0 ,  6.2  ), ( 4.3 , 6.2 )),
(( 0.1 ,  6.1  ), ( 4.3 , 6.1 )),
(( 0.1 ,  6.0  ), ( 4.2 , 6.0 )),
(( 0.1 ,  5.9  ), ( 4.2 , 5.9 )),
(( 0.1 ,  5.8  ), ( 4.1 , 5.8 )),
(( 0.1 ,  5.7  ), ( 0.6 , 5.7 )),
(( 3.6 ,  5.7  ), ( 4.1 , 5.7 )),
(( 0.1 ,  5.6  ), ( 0.5 , 5.6 )),
(( 3.5 ,  5.6  ), ( 4.0 , 5.6 )),
(( 0.1 ,  5.5  ), ( 0.5 , 5.5 )),
(( 3.5 ,  5.5  ), ( 4.0 , 5.5 )),
(( 0.1 ,  5.4  ), ( 0.4 , 5.4 )),
(( 3.4 ,  5.4  ), ( 3.9 , 5.4 )),
(( 0.1 ,  5.3  ), ( 0.4 , 5.3 )),
(( 3.4 ,  5.3  ), ( 3.8 , 5.3 )),
(( 0.1 ,  5.2  ), ( 0.4 , 5.2 )),
(( 3.3 ,  5.2  ), ( 3.8 , 5.2 )),
(( 0.1 ,  5.1  ), ( 0.4 , 5.1 )),
(( 3.3 ,  5.1  ), ( 3.7 , 5.1 )),
(( 0.1 ,  5.0  ), ( 0.4 , 5.0 )),
(( 3.2 ,  5.0  ), ( 3.7 , 5.0 )),
(( 0.1 ,  4.9  ), ( 0.4 , 4.9 )),
(( 3.1 ,  4.9  ), ( 3.6 , 4.9 )),
(( 0.1 ,  4.8  ), ( 0.4 , 4.8 )),
(( 3.1 ,  4.8  ), ( 3.6 , 4.8 )),
(( 0.1 ,  4.7  ), ( 0.4 , 4.7 )),
(( 3.0 ,  4.7  ), ( 3.5 , 4.7 )),
(( 0.0 ,  4.6  ), ( 0.3 , 4.6 )),
(( 3.0 ,  4.6  ), ( 3.5 , 4.6 )),
(( 0.0 ,  4.5  ), ( 0.3 , 4.5 )),
(( 2.9 ,  4.5  ), ( 3.4 , 4.5 )),
(( 2.9 ,  4.4  ), ( 3.3 , 4.4 )),
(( 2.8 ,  4.3  ), ( 3.3 , 4.3 )),
(( 2.7 ,  4.2  ), ( 3.2 , 4.2 )),
(( 2.7 ,  4.1  ), ( 3.2 , 4.1 )),
(( 2.6 ,  4.0  ), ( 3.1 , 4.0 )),
(( 2.6 ,  3.9  ), ( 3.1 , 3.9 )),
(( 2.5 ,  3.8  ), ( 3.0 , 3.8 )),
(( 2.4 ,  3.7  ), ( 3.0 , 3.7 )),
(( 2.4 ,  3.6  ), ( 2.9 , 3.6 )),
(( 2.3 ,  3.5  ), ( 2.9 , 3.5 )),
(( 2.3 ,  3.4  ), ( 2.8 , 3.4 )),
(( 2.2 ,  3.3  ), ( 2.8 , 3.3 )),
(( 2.1 ,  3.2  ), ( 2.7 , 3.2 )),
(( 2.1 ,  3.1  ), ( 2.7 , 3.1 )),
(( 2.0 ,  3.0  ), ( 2.6 , 3.0 )),
(( 2.0 ,  2.9  ), ( 2.6 , 2.9 )),
(( 1.9 ,  2.8  ), ( 2.5 , 2.8 )),
(( 1.8 ,  2.7  ), ( 2.5 , 2.7 )),
(( 1.8 ,  2.6  ), ( 2.4 , 2.6 )),
(( 1.7 ,  2.5  ), ( 2.4 , 2.5 )),
(( 1.7 ,  2.4  ), ( 2.3 , 2.4 )),
(( 1.6 ,  2.3  ), ( 2.3 , 2.3 )),
(( 1.5 ,  2.2  ), ( 2.2 , 2.2 )),
(( 1.5 ,  2.1  ), ( 2.2 , 2.1 )),
(( 1.4 ,  2.0  ), ( 2.1 , 2.0 )),
(( 1.4 ,  1.9  ), ( 2.1 , 1.9 )),
(( 1.3 ,  1.8  ), ( 2.0 , 1.8 )),
(( 1.2 ,  1.7  ), ( 2.0 , 1.7 )),
(( 1.2 ,  1.6  ), ( 1.9 , 1.6 )),
(( 1.1 ,  1.5  ), ( 1.9 , 1.5 )),
(( 1.0 ,  1.4  ), ( 1.8 , 1.4 )),
(( 1.0 ,  1.3  ), ( 1.8 , 1.3 )),
(( 0.9 ,  1.2  ), ( 1.7 , 1.2 )),
(( 0.9 ,  1.1  ), ( 1.7 , 1.1 )),
(( 0.8 ,  1.0  ), ( 1.6 , 1.0 )),
(( 0.7 ,  0.9  ), ( 1.6 , 0.9 )),
(( 0.7 ,  0.8  ), ( 1.6 , 0.8 )),
(( 0.6 ,  0.7  ), ( 1.5 , 0.7 )),
(( 0.6 ,  0.6  ), ( 1.5 , 0.6 )),
(( 0.5 ,  0.5  ), ( 1.4 , 0.5 )),
(( 0.4 ,  0.4  ), ( 1.4 , 0.4 )),
(( 0.4 ,  0.3  ), ( 1.3 , 0.3 )),
(( 0.3 ,  0.2  ), ( 1.3 , 0.2 )),
(( 0.2 ,  0.1  ), ( 1.3 , 0.1 )),
(( 0.3 ,  0.0  ), ( 1.2 , 0.0 ))],
        '8': [(( 1.7 ,  6.6  ), ( 2.8 , 6.6 )),
(( 1.4 ,  6.5  ), ( 3.1 , 6.5 )),
(( 1.1 ,  6.4  ), ( 3.3 , 6.4 )),
(( 1.0 ,  6.3  ), ( 1.7 , 6.3 )),
(( 2.6 ,  6.3  ), ( 3.5 , 6.3 )),
(( 0.9 ,  6.2  ), ( 1.5 , 6.2 )),
(( 2.8 ,  6.2  ), ( 3.6 , 6.2 )),
(( 0.7 ,  6.1  ), ( 1.4 , 6.1 )),
(( 2.9 ,  6.1  ), ( 3.7 , 6.1 )),
(( 0.7 ,  6.0  ), ( 1.3 , 6.0 )),
(( 3.0 ,  6.0  ), ( 3.8 , 6.0 )),
(( 0.6 ,  5.9  ), ( 1.2 , 5.9 )),
(( 3.1 ,  5.9  ), ( 3.8 , 5.9 )),
(( 0.5 ,  5.8  ), ( 1.1 , 5.8 )),
(( 3.2 ,  5.8  ), ( 3.9 , 5.8 )),
(( 0.5 ,  5.7  ), ( 1.1 , 5.7 )),
(( 3.2 ,  5.7  ), ( 3.9 , 5.7 )),
(( 0.4 ,  5.6  ), ( 1.0 , 5.6 )),
(( 3.2 ,  5.6  ), ( 3.9 , 5.6 )),
(( 0.4 ,  5.5  ), ( 1.0 , 5.5 )),
(( 3.3 ,  5.5  ), ( 4.0 , 5.5 )),
(( 0.3 ,  5.4  ), ( 1.0 , 5.4 )),
(( 3.3 ,  5.4  ), ( 4.0 , 5.4 )),
(( 0.3 ,  5.3  ), ( 1.0 , 5.3 )),
(( 3.3 ,  5.3  ), ( 4.0 , 5.3 )),
(( 0.3 ,  5.2  ), ( 1.0 , 5.2 )),
(( 3.3 ,  5.2  ), ( 4.0 , 5.2 )),
(( 0.3 ,  5.1  ), ( 1.0 , 5.1 )),
(( 3.3 ,  5.1  ), ( 4.0 , 5.1 )),
(( 0.3 ,  5.0  ), ( 1.1 , 5.0 )),
(( 3.3 ,  5.0  ), ( 3.9 , 5.0 )),
(( 0.3 ,  4.9  ), ( 1.1 , 4.9 )),
(( 3.3 ,  4.9  ), ( 3.9 , 4.9 )),
(( 0.3 ,  4.8  ), ( 1.1 , 4.8 )),
(( 3.3 ,  4.8  ), ( 3.9 , 4.8 )),
(( 0.3 ,  4.7  ), ( 1.2 , 4.7 )),
(( 3.2 ,  4.7  ), ( 3.8 , 4.7 )),
(( 0.3 ,  4.6  ), ( 1.3 , 4.6 )),
(( 3.2 ,  4.6  ), ( 3.8 , 4.6 )),
(( 0.3 ,  4.5  ), ( 1.4 , 4.5 )),
(( 3.1 ,  4.5  ), ( 3.7 , 4.5 )),
(( 0.4 ,  4.4  ), ( 1.5 , 4.4 )),
(( 3.1 ,  4.4  ), ( 3.6 , 4.4 )),
(( 0.4 ,  4.3  ), ( 1.6 , 4.3 )),
(( 3.0 ,  4.3  ), ( 3.5 , 4.3 )),
(( 0.5 ,  4.2  ), ( 1.8 , 4.2 )),
(( 2.9 ,  4.2  ), ( 3.4 , 4.2 )),
(( 0.5 ,  4.1  ), ( 2.0 , 4.1 )),
(( 2.8 ,  4.1  ), ( 3.3 , 4.1 )),
(( 0.6 ,  4.0  ), ( 2.2 , 4.0 )),
(( 2.6 ,  4.0  ), ( 3.2 , 4.0 )),
(( 0.7 ,  3.9  ), ( 2.4 , 3.9 )),
(( 2.5 ,  3.9  ), ( 3.0 , 3.9 )),
(( 0.8 ,  3.8  ), ( 2.9 , 3.8 )),
(( 0.9 ,  3.7  ), ( 2.9 , 3.7 )),
(( 1.1 ,  3.6  ), ( 3.0 , 3.6 )),
(( 1.2 ,  3.5  ), ( 3.2 , 3.5 )),
(( 1.3 ,  3.4  ), ( 3.4 , 3.4 )),
(( 1.2 ,  3.3  ), ( 3.6 , 3.3 )),
(( 1.0 ,  3.2  ), ( 1.6 , 3.2 )),
(( 1.9 ,  3.2  ), ( 3.7 , 3.2 )),
(( 0.9 ,  3.1  ), ( 1.5 , 3.1 )),
(( 2.1 ,  3.1  ), ( 3.8 , 3.1 )),
(( 0.7 ,  3.0  ), ( 1.3 , 3.0 )),
(( 2.3 ,  3.0  ), ( 3.9 , 3.0 )),
(( 0.6 ,  2.9  ), ( 1.2 , 2.9 )),
(( 2.5 ,  2.9  ), ( 4.0 , 2.9 )),
(( 0.5 ,  2.8  ), ( 1.1 , 2.8 )),
(( 2.7 ,  2.8  ), ( 4.1 , 2.8 )),
(( 0.4 ,  2.7  ), ( 1.0 , 2.7 )),
(( 2.8 ,  2.7  ), ( 4.1 , 2.7 )),
(( 0.3 ,  2.6  ), ( 1.0 , 2.6 )),
(( 3.0 ,  2.6  ), ( 4.2 , 2.6 )),
(( 0.3 ,  2.5  ), ( 0.9 , 2.5 )),
(( 3.1 ,  2.5  ), ( 4.2 , 2.5 )),
(( 0.2 ,  2.4  ), ( 0.9 , 2.4 )),
(( 3.2 ,  2.4  ), ( 4.2 , 2.4 )),
(( 0.2 ,  2.3  ), ( 0.8 , 2.3 )),
(( 3.3 ,  2.3  ), ( 4.2 , 2.3 )),
(( 0.1 ,  2.2  ), ( 0.8 , 2.2 )),
(( 3.4 ,  2.2  ), ( 4.2 , 2.2 )),
(( 0.1 ,  2.1  ), ( 0.8 , 2.1 )),
(( 3.4 ,  2.1  ), ( 4.2 , 2.1 )),
(( 0.1 ,  2.0  ), ( 0.7 , 2.0 )),
(( 3.4 ,  2.0  ), ( 4.2 , 2.0 )),
(( 0.1 ,  1.9  ), ( 0.7 , 1.9 )),
(( 3.5 ,  1.9  ), ( 4.2 , 1.9 )),
(( 0.1 ,  1.8  ), ( 0.7 , 1.8 )),
(( 3.5 ,  1.8  ), ( 4.2 , 1.8 )),
(( 0.0 ,  1.7  ), ( 0.7 , 1.7 )),
(( 3.5 ,  1.7  ), ( 4.2 , 1.7 )),
(( 0.0 ,  1.6  ), ( 0.7 , 1.6 )),
(( 3.5 ,  1.6  ), ( 4.2 , 1.6 )),
(( 0.1 ,  1.5  ), ( 0.7 , 1.5 )),
(( 3.5 ,  1.5  ), ( 4.2 , 1.5 )),
(( 0.1 ,  1.4  ), ( 0.7 , 1.4 )),
(( 3.5 ,  1.4  ), ( 4.1 , 1.4 )),
(( 0.1 ,  1.3  ), ( 0.8 , 1.3 )),
(( 3.5 ,  1.3  ), ( 4.1 , 1.3 )),
(( 0.1 ,  1.2  ), ( 0.8 , 1.2 )),
(( 3.4 ,  1.2  ), ( 4.0 , 1.2 )),
(( 0.1 ,  1.1  ), ( 0.8 , 1.1 )),
(( 3.4 ,  1.1  ), ( 4.0 , 1.1 )),
(( 0.2 ,  1.0  ), ( 0.8 , 1.0 )),
(( 3.4 ,  1.0  ), ( 3.9 , 1.0 )),
(( 0.2 ,  0.9  ), ( 0.9 , 0.9 )),
(( 3.3 ,  0.9  ), ( 3.9 , 0.9 )),
(( 0.3 ,  0.8  ), ( 1.0 , 0.8 )),
(( 3.3 ,  0.8  ), ( 3.8 , 0.8 )),
(( 0.3 ,  0.7  ), ( 1.0 , 0.7 )),
(( 3.2 ,  0.7  ), ( 3.7 , 0.7 )),
(( 0.4 ,  0.6  ), ( 1.1 , 0.6 )),
(( 3.1 ,  0.6  ), ( 3.6 , 0.6 )),
(( 0.5 ,  0.5  ), ( 1.3 , 0.5 )),
(( 3.0 ,  0.5  ), ( 3.5 , 0.5 )),
(( 0.6 ,  0.4  ), ( 1.4 , 0.4 )),
(( 2.8 ,  0.4  ), ( 3.4 , 0.4 )),
(( 0.7 ,  0.3  ), ( 1.6 , 0.3 )),
(( 2.5 ,  0.3  ), ( 3.2 , 0.3 )),
(( 0.9 ,  0.2  ), ( 3.0 , 0.2 )),
(( 1.1 ,  0.1  ), ( 2.8 , 0.1 )),
(( 1.4 ,  0.0  ), ( 2.4 , 0.0 ))],
        '9': [(( 1.6 ,  6.6  ), ( 2.6 , 6.6 )),
(( 1.3 ,  6.5  ), ( 3.0 , 6.5 )),
(( 1.1 ,  6.4  ), ( 3.2 , 6.4 )),
(( 0.9 ,  6.3  ), ( 1.7 , 6.3 )),
(( 2.4 ,  6.3  ), ( 3.3 , 6.3 )),
(( 0.8 ,  6.2  ), ( 1.5 , 6.2 )),
(( 2.6 ,  6.2  ), ( 3.5 , 6.2 )),
(( 0.7 ,  6.1  ), ( 1.3 , 6.1 )),
(( 2.8 ,  6.1  ), ( 3.6 , 6.1 )),
(( 0.6 ,  6.0  ), ( 1.2 , 6.0 )),
(( 2.9 ,  6.0  ), ( 3.7 , 6.0 )),
(( 0.5 ,  5.9  ), ( 1.1 , 5.9 )),
(( 2.9 ,  5.9  ), ( 3.7 , 5.9 )),
(( 0.4 ,  5.8  ), ( 1.1 , 5.8 )),
(( 3.0 ,  5.8  ), ( 3.8 , 5.8 )),
(( 0.3 ,  5.7  ), ( 1.0 , 5.7 )),
(( 3.1 ,  5.7  ), ( 3.9 , 5.7 )),
(( 0.3 ,  5.6  ), ( 1.0 , 5.6 )),
(( 3.1 ,  5.6  ), ( 3.9 , 5.6 )),
(( 0.2 ,  5.5  ), ( 0.9 , 5.5 )),
(( 3.2 ,  5.5  ), ( 4.0 , 5.5 )),
(( 0.2 ,  5.4  ), ( 0.9 , 5.4 )),
(( 3.2 ,  5.4  ), ( 4.0 , 5.4 )),
(( 0.1 ,  5.3  ), ( 0.9 , 5.3 )),
(( 3.2 ,  5.3  ), ( 4.1 , 5.3 )),
(( 0.1 ,  5.2  ), ( 0.8 , 5.2 )),
(( 3.3 ,  5.2  ), ( 4.1 , 5.2 )),
(( 0.1 ,  5.1  ), ( 0.8 , 5.1 )),
(( 3.3 ,  5.1  ), ( 4.1 , 5.1 )),
(( 0.0 ,  5.0  ), ( 0.8 , 5.0 )),
(( 3.3 ,  5.0  ), ( 4.1 , 5.0 )),
(( 0.0 ,  4.9  ), ( 0.8 , 4.9 )),
(( 3.3 ,  4.9  ), ( 4.2 , 4.9 )),
(( 0.0 ,  4.8  ), ( 0.8 , 4.8 )),
(( 3.3 ,  4.8  ), ( 4.2 , 4.8 )),
(( 0.0 ,  4.7  ), ( 0.8 , 4.7 )),
(( 3.3 ,  4.7  ), ( 4.2 , 4.7 )),
(( 0.0 ,  4.6  ), ( 0.8 , 4.6 )),
(( 3.3 ,  4.6  ), ( 4.2 , 4.6 )),
(( 0.0 ,  4.5  ), ( 0.8 , 4.5 )),
(( 3.3 ,  4.5  ), ( 4.2 , 4.5 )),
(( 0.0 ,  4.4  ), ( 0.8 , 4.4 )),
(( 3.3 ,  4.4  ), ( 4.2 , 4.4 )),
(( 0.0 ,  4.3  ), ( 0.8 , 4.3 )),
(( 3.3 ,  4.3  ), ( 4.2 , 4.3 )),
(( 0.0 ,  4.2  ), ( 0.8 , 4.2 )),
(( 3.3 ,  4.2  ), ( 4.2 , 4.2 )),
(( 0.0 ,  4.1  ), ( 0.9 , 4.1 )),
(( 3.3 ,  4.1  ), ( 4.2 , 4.1 )),
(( 0.0 ,  4.0  ), ( 0.9 , 4.0 )),
(( 3.3 ,  4.0  ), ( 4.2 , 4.0 )),
(( 0.0 ,  3.9  ), ( 0.9 , 3.9 )),
(( 3.3 ,  3.9  ), ( 4.2 , 3.9 )),
(( 0.0 ,  3.8  ), ( 1.0 , 3.8 )),
(( 3.2 ,  3.8  ), ( 4.2 , 3.8 )),
(( 0.1 ,  3.7  ), ( 1.0 , 3.7 )),
(( 3.2 ,  3.7  ), ( 4.2 , 3.7 )),
(( 0.1 ,  3.6  ), ( 1.1 , 3.6 )),
(( 3.1 ,  3.6  ), ( 4.2 , 3.6 )),
(( 0.1 ,  3.5  ), ( 1.2 , 3.5 )),
(( 3.1 ,  3.5  ), ( 4.2 , 3.5 )),
(( 0.2 ,  3.4  ), ( 1.2 , 3.4 )),
(( 3.0 ,  3.4  ), ( 4.2 , 3.4 )),
(( 0.2 ,  3.3  ), ( 1.3 , 3.3 )),
(( 2.9 ,  3.3  ), ( 4.2 , 3.3 )),
(( 0.3 ,  3.2  ), ( 1.5 , 3.2 )),
(( 2.7 ,  3.2  ), ( 3.2 , 3.2 )),
(( 3.3 ,  3.2  ), ( 4.2 , 3.2 )),
(( 0.4 ,  3.1  ), ( 1.7 , 3.1 )),
(( 2.5 ,  3.1  ), ( 3.0 , 3.1 )),
(( 3.3 ,  3.1  ), ( 4.1 , 3.1 )),
(( 0.4 ,  3.0  ), ( 2.9 , 3.0 )),
(( 3.2 ,  3.0  ), ( 4.1 , 3.0 )),
(( 0.5 ,  2.9  ), ( 2.8 , 2.9 )),
(( 3.2 ,  2.9  ), ( 4.1 , 2.9 )),
(( 0.6 ,  2.8  ), ( 2.7 , 2.8 )),
(( 3.2 ,  2.8  ), ( 4.1 , 2.8 )),
(( 0.8 ,  2.7  ), ( 2.5 , 2.7 )),
(( 3.2 ,  2.7  ), ( 4.0 , 2.7 )),
(( 1.0 ,  2.6  ), ( 2.4 , 2.6 )),
(( 3.2 ,  2.6  ), ( 4.0 , 2.6 )),
(( 1.3 ,  2.5  ), ( 2.1 , 2.5 )),
(( 3.1 ,  2.5  ), ( 4.0 , 2.5 )),
(( 3.1 ,  2.4  ), ( 3.9 , 2.4 )),
(( 3.1 ,  2.3  ), ( 3.9 , 2.3 )),
(( 3.0 ,  2.2  ), ( 3.8 , 2.2 )),
(( 3.0 ,  2.1  ), ( 3.8 , 2.1 )),
(( 2.9 ,  2.0  ), ( 3.7 , 2.0 )),
(( 2.9 ,  1.9  ), ( 3.6 , 1.9 )),
(( 2.8 ,  1.8  ), ( 3.6 , 1.8 )),
(( 2.8 ,  1.7  ), ( 3.5 , 1.7 )),
(( 2.7 ,  1.6  ), ( 3.4 , 1.6 )),
(( 2.6 ,  1.5  ), ( 3.4 , 1.5 )),
(( 2.5 ,  1.4  ), ( 3.3 , 1.4 )),
(( 2.4 ,  1.3  ), ( 3.2 , 1.3 )),
(( 2.3 ,  1.2  ), ( 3.1 , 1.2 )),
(( 2.2 ,  1.1  ), ( 3.0 , 1.1 )),
(( 2.1 ,  1.0  ), ( 2.9 , 1.0 )),
(( 2.0 ,  0.9  ), ( 2.8 , 0.9 )),
(( 1.9 ,  0.8  ), ( 2.6 , 0.8 )),
(( 1.7 ,  0.7  ), ( 2.5 , 0.7 )),
(( 1.5 ,  0.6  ), ( 2.3 , 0.6 )),
(( 1.3 ,  0.5  ), ( 2.2 , 0.5 )),
(( 1.1 ,  0.4  ), ( 2.0 , 0.4 )),
(( 0.8 ,  0.3  ), ( 1.8 , 0.3 )),
(( 0.4 ,  0.2  ), ( 1.6 , 0.2 )),
(( 0.4 ,  0.1  ), ( 1.3 , 0.1 )),
(( 0.6 ,  0.0  ), ( 0.9 , 0.0 ))],
'.': [(( 0.3 ,  1.0  ), ( 0.7 , 1.0 )),
(( 0.1 ,  0.9  ), ( 0.9 , 0.9 )),
(( 0.0 ,  0.8  ), ( 1.0 , 0.8 )),
(( 0.0 ,  0.7  ), ( 1.0 , 0.7 )),
(( 0.0 ,  0.6  ), ( 1.1 , 0.6 )),
(( 0.0 ,  0.5  ), ( 1.1 , 0.5 )),
(( 0.0 ,  0.4  ), ( 1.0 , 0.4 )),
(( 0.0 ,  0.3  ), ( 1.0 , 0.3 )),
(( 0.0 ,  0.2  ), ( 1.0 , 0.2 )),
(( 0.1 ,  0.1  ), ( 0.9 , 0.1 )),
(( 0.3 ,  0.0  ), ( 0.7 , 0.0 ))],
'/': [(( 3.4 ,  8.2  ), ( 4.1 , 8.2 )),
(( 3.3 ,  8.1  ), ( 4.0 , 8.1 )),
(( 3.3 ,  8.0  ), ( 4.0 , 8.0 )),
(( 3.3 ,  7.9  ), ( 3.9 , 7.9 )),
(( 3.2 ,  7.8  ), ( 3.9 , 7.8 )),
(( 3.2 ,  7.7  ), ( 3.8 , 7.7 )),
(( 3.1 ,  7.6  ), ( 3.8 , 7.6 )),
(( 3.1 ,  7.5  ), ( 3.8 , 7.5 )),
(( 3.1 ,  7.4  ), ( 3.7 , 7.4 )),
(( 3.0 ,  7.3  ), ( 3.7 , 7.3 )),
(( 3.0 ,  7.2  ), ( 3.6 , 7.2 )),
(( 2.9 ,  7.1  ), ( 3.6 , 7.1 )),
(( 2.9 ,  7.0  ), ( 3.6 , 7.0 )),
(( 2.9 ,  6.9  ), ( 3.5 , 6.9 )),
(( 2.8 ,  6.8  ), ( 3.5 , 6.8 )),
(( 2.8 ,  6.7  ), ( 3.4 , 6.7 )),
(( 2.7 ,  6.6  ), ( 3.4 , 6.6 )),
(( 2.7 ,  6.5  ), ( 3.3 , 6.5 )),
(( 2.6 ,  6.4  ), ( 3.3 , 6.4 )),
(( 2.6 ,  6.3  ), ( 3.3 , 6.3 )),
(( 2.6 ,  6.2  ), ( 3.2 , 6.2 )),
(( 2.5 ,  6.1  ), ( 3.2 , 6.1 )),
(( 2.5 ,  6.0  ), ( 3.1 , 6.0 )),
(( 2.4 ,  5.9  ), ( 3.1 , 5.9 )),
(( 2.4 ,  5.8  ), ( 3.1 , 5.8 )),
(( 2.4 ,  5.7  ), ( 3.0 , 5.7 )),
(( 2.3 ,  5.6  ), ( 3.0 , 5.6 )),
(( 2.3 ,  5.5  ), ( 2.9 , 5.5 )),
(( 2.2 ,  5.4  ), ( 2.9 , 5.4 )),
(( 2.2 ,  5.3  ), ( 2.8 , 5.3 )),
(( 2.1 ,  5.2  ), ( 2.8 , 5.2 )),
(( 2.1 ,  5.1  ), ( 2.8 , 5.1 )),
(( 2.1 ,  5.0  ), ( 2.7 , 5.0 )),
(( 2.0 ,  4.9  ), ( 2.7 , 4.9 )),
(( 2.0 ,  4.8  ), ( 2.6 , 4.8 )),
(( 1.9 ,  4.7  ), ( 2.6 , 4.7 )),
(( 1.9 ,  4.6  ), ( 2.6 , 4.6 )),
(( 1.9 ,  4.5  ), ( 2.5 , 4.5 )),
(( 1.8 ,  4.4  ), ( 2.5 , 4.4 )),
(( 1.8 ,  4.3  ), ( 2.4 , 4.3 )),
(( 1.7 ,  4.2  ), ( 2.4 , 4.2 )),
(( 1.7 ,  4.1  ), ( 2.3 , 4.1 )),
(( 1.6 ,  4.0  ), ( 2.3 , 4.0 )),
(( 1.6 ,  3.9  ), ( 2.3 , 3.9 )),
(( 1.6 ,  3.8  ), ( 2.2 , 3.8 )),
(( 1.5 ,  3.7  ), ( 2.2 , 3.7 )),
(( 1.5 ,  3.6  ), ( 2.1 , 3.6 )),
(( 1.4 ,  3.5  ), ( 2.1 , 3.5 )),
(( 1.4 ,  3.4  ), ( 2.1 , 3.4 )),
(( 1.4 ,  3.3  ), ( 2.0 , 3.3 )),
(( 1.3 ,  3.2  ), ( 2.0 , 3.2 )),
(( 1.3 ,  3.1  ), ( 1.9 , 3.1 )),
(( 1.2 ,  3.0  ), ( 1.9 , 3.0 )),
(( 1.2 ,  2.9  ), ( 1.8 , 2.9 )),
(( 1.1 ,  2.8  ), ( 1.8 , 2.8 )),
(( 1.1 ,  2.7  ), ( 1.8 , 2.7 )),
(( 1.1 ,  2.6  ), ( 1.7 , 2.6 )),
(( 1.0 ,  2.5  ), ( 1.7 , 2.5 )),
(( 1.0 ,  2.4  ), ( 1.6 , 2.4 )),
(( 0.9 ,  2.3  ), ( 1.6 , 2.3 )),
(( 0.9 ,  2.2  ), ( 1.5 , 2.2 )),
(( 0.9 ,  2.1  ), ( 1.5 , 2.1 )),
(( 0.8 ,  2.0  ), ( 1.5 , 2.0 )),
(( 0.8 ,  1.9  ), ( 1.4 , 1.9 )),
(( 0.7 ,  1.8  ), ( 1.4 , 1.8 )),
(( 0.7 ,  1.7  ), ( 1.3 , 1.7 )),
(( 0.6 ,  1.6  ), ( 1.3 , 1.6 )),
(( 0.6 ,  1.5  ), ( 1.3 , 1.5 )),
(( 0.6 ,  1.4  ), ( 1.2 , 1.4 )),
(( 0.5 ,  1.3  ), ( 1.2 , 1.3 )),
(( 0.5 ,  1.2  ), ( 1.1 , 1.2 )),
(( 0.4 ,  1.1  ), ( 1.1 , 1.1 )),
(( 0.4 ,  1.0  ), ( 1.0 , 1.0 )),
(( 0.4 ,  0.9  ), ( 1.0 , 0.9 )),
(( 0.3 ,  0.8  ), ( 1.0 , 0.8 )),
(( 0.3 ,  0.7  ), ( 0.9 , 0.7 )),
(( 0.2 ,  0.6  ), ( 0.9 , 0.6 )),
(( 0.2 ,  0.5  ), ( 0.8 , 0.5 )),
(( 0.2 ,  0.4  ), ( 0.8 , 0.4 )),
(( 0.1 ,  0.3  ), ( 0.8 , 0.3 )),
(( 0.1 ,  0.2  ), ( 0.7 , 0.2 )),
(( 0.0 ,  0.1  ), ( 0.7 , 0.1 )),
(( 0.0 ,  0.0  ), ( 0.6 , 0.0 ))],
'-': [(( 0.2 ,  0.6  ), ( 2.8 , 0.6 )),
(( 0.2 ,  0.5  ), ( 2.8 , 0.5 )),
(( 0.2 ,  0.4  ), ( 2.7 , 0.4 )),
(( 0.1 ,  0.3  ), ( 2.7 , 0.3 )),
(( 0.1 ,  0.2  ), ( 2.7 , 0.2 )),
(( 0.0 ,  0.1  ), ( 2.6 , 0.1 )),
(( 0.0 ,  0.0  ), ( 2.6 , 0.0 ))]
    }

chars = {'1': Polygon([(3.5, 8), (0, 8), (0, 0), (14, 0), (14, 8), (11, 8), (11, 35), (5, 35), (0, 28), (0, 23), (3.5, 23)]),
         '2':Polygon([(0, 8), (0, 0), (22, 0), (22,9), (13, 9), (22, 20), (22, 29), (16.5, 35), (5.5, 35), (0, 29), (0, 22), (9, 22), (9, 26), (10, 27), (13, 27), (14, 26), (14, 23)]),
         '3': Polygon([(0, 6), (6, 0), (20, 0), (26,6), (26, 17), (24.5, 18), (26, 19), (26, 29), (20, 35), (6, 35), (0, 29), (0, 23), (9.5, 23), (9.5, 26), (10.5, 27), (15.5, 27), (16.5, 26), (16.5, 22), (15.5, 21), (11, 21), (11, 14), (15.5, 14), (16.5, 13), (16.5, 9.5), (15.5, 8.5), (10, 8.5), (9, 9.5), (9, 11.5), (0, 11.5)]),
         '4':Polygon([(10, 7), (10, 0), (25, 0), (25,7), (22, 7), (22, 11), (25, 11), (25, 18), (22, 18), (22, 36), (14, 36), (0, 21), (0, 11), (13, 11), (13, 7)],holes=[[(8,18),(14,18),(14,24)]]),
         '5': Polygon([(9, 12), (0, 12), (0, 6.5), (6.5,0), (21, 0), (27, 6), (27, 19), (21, 24), (10.5, 24), (9.5, 25), (9.5, 27), (27, 27), (27, 35), (0, 35), (0, 22), (6, 16.5), (16, 16.5), (17.5, 15), (17.5, 8), (16, 6.5), (10.5, 6.5), (9, 8)]),
         '6':Polygon([(0, 6), (6, 0), (20, 0), (26,6), (26, 15), (19.5, 20.5), (8, 20.5), (8, 25), (10, 28), (16, 28), (18, 25),(18, 23), (26, 23), (26, 28), (21, 35), (5, 35), (0, 28)],holes=[[(8,8),(10,6),(16,6),(18,8),(18,13),(16,15),(10,15),(8,13)]]),
         '7':Polygon([(5, 0), (14, 0), (25, 27), (25,34), (0, 34), (0, 23.5), (6.5, 23.5), (6.5, 26), (15, 26)]),
         '8':Polygon([(0, 5), (5, 0), (19, 0), (24,5), (24, 15), (21.5, 17), (24, 19), (24, 30), (19, 35), (5, 35), (0, 30),(0, 19), (2.5, 17), (0, 15)],holes=[[(7,8),(9,6),(15,6),(17,8),(17,13),(15,15),(9,15),(7,13)],[(7,22),(9,20),(15,20),(17,22),(17,27),(15,29),(9,29),(7,27)]]),
         '9':Polygon([(26, 29), (20, 35), (6, 35), (0,29), (0, 20), (6.5, 14.5), (18, 14.5), (18, 10), (16, 7), (10, 7), (8, 10),(8, 12), (0, 12), (0, 7), (5, 0), (21, 0), (26, 7)],holes=[[(18,27),(16,29),(10,29),(8,27),(8,22),(10,20),(16,20),(18,22)]]),
         '0':Polygon([(0, 5), (5, 0), (19, 0), (24,5), (24, 30), (19, 35), (5, 35), (0, 30)],holes=[[(7,8),(9,6),(15,6),(17,8),(17,27),(15,29),(9,29),(7,27)]]),
         '.':Polygon([(0, 1), (1, 0), (3, 0), (4,1), (4, 3), (3, 4), (1, 4), (0, 3)]),
         '/':Polygon([(0, 3),(0, 1), (1, 0), (3, 0),  (25, 32), (25, 34), (24, 35), (22, 35)])}

char_shifts = {'1':16,
               '2':26,
               '3':28,
               '4':28,
               '5':29,
               '6':28,
               '7':28,
               '8':27,
               '9':28,
               '0':28,
               '.':4,
               '/':27}

def rasberitesb(sender,app_data):
    if app_data:
        if sender == 'change_order':
            dpg.set_value('add_text',False)
            dpg.set_value('movelines',False)
        if sender == 'add_text':
            dpg.set_value('change_order',False)
            dpg.set_value('movelines',False)
        if sender == 'movelines':
            dpg.set_value('change_order',False)
            dpg.set_value('add_text',False)

def plot_mouse_click_callback():
    
    x,y = dpg.get_plot_mouse_pos()
    if dpg.get_value('change_order'):

        rec = db.get_records('lines')

        lines = [[(row[1],row[2]),(row[3],row[4])] for row in rec]


        l = find_closest_lines(lines,(x,y),range(len(lines)))###################
        
        if dpg.get_value('color_1'):
            db.set_color_where_id('lines',0,l)
        elif dpg.get_value('color_2'):
            db.set_color_where_id('lines',1,l)
        elif dpg.get_value('color_3'):
            db.set_color_where_id('lines',2,l)
        elif dpg.get_value('color_4'):
            db.set_color_where_id('lines',3,l)
        elif dpg.get_value('color_5'):
            db.set_color_where_id('lines',4,l)
        redraw()
    elif dpg.get_value('add_text'):
        delta = 0
        val = dpg.get_value('insert_numbers')
        nice_path = 'n'+val
        iter = 1
        while 1:
            for i in db.get_unique_values('lines','parent'):
                if i == nice_path:
                    nice_path = 'n'+val + f' (copy {iter})'
                    iter +=1
            else: 
                break
        dpg.add_button(label=nice_path,parent='butonss',tag=nice_path,callback=active_but)
        lines = []
        heig = 10
        kf = 35/heig

        for ch in val:
            # for l in digit_lines[ch]:
            #     db.add_record('lines', (round(x+l[0][0] + delta,4),  round(y+l[0][1],4),round(x+l[1][0] + delta,4), round(y+l[1][1],4),0,nice_path,0,1))
            polygon2 = chars[ch]
            
            while 1:
                if polygon2.geom_type == 'Polygon':
                    x1, y1 = polygon2.exterior.xy
                    
                    
                    for h in range(len(x1)-1):
                        lines.append((round(x1[h]/kf+x+delta,4),  round(y1[h]/kf+y,4), round(x1[h+1]/kf+x+delta,4), round(y1[h+1]/kf+y,4),0,nice_path,0,1))
                    for p in polygon2.interiors:
                        x1, y1 = p.xy
                        for h in range(len(x1)-1):
                            lines.append((round(x1[h]/kf+x+delta,4),  round(y1[h]/kf+y,4), round(x1[h+1]/kf+x+delta,4), round(y1[h+1]/kf+y,4),0,nice_path,0,1))
                    polygon2 = polygon2.buffer(-0.2*kf,quad_segs=0)
                elif polygon2.geom_type == 'MultiPolygon':
                    for pol in polygon2.geoms:
                        x1, y1 = pol.exterior.xy
                        for h in range(len(x1)-1):
                            lines.append((round(x1[h]/kf+x+delta,4),  round(y1[h]/kf+y,4), round(x1[h+1]/kf+x+delta,4), round(y1[h+1]/kf+y,4),0,nice_path,0,1))
                        for p in pol.interiors:
                            x1, y1 = p.xy
                            for h in range(len(x1)-1):
                                lines.append((round(x1[h]/kf+x+delta,4),  round(y1[h]/kf+y,4), round(x1[h+1]/kf+x+delta,4), round(y1[h+1]/kf+y,4),0,nice_path,0,1))
                    polygon2 = polygon2.buffer(-0.2*kf,quad_segs=0)
                if polygon2.is_empty:
                    break
            delta += char_shifts[ch]/kf
        db.add_multiple_records('lines',lines)
        redraw()
    elif dpg.get_value('movelines'):
        rec = db.get_records_where('lines','isactive=1')
        xx = [r[1] for r in rec]
        xx+= [r[3] for r in rec]
        yy = [r[2] for r in rec]
        yy+= [r[4] for r in rec]
        db.increment_field_value_with_condition('lines','sx','ex','sy','ey',x-min(xx),y-min(yy),'isactive',1)
        redraw()
def recolor():
    rec = db.get_records('lines')
    for l in rec:
        
        if l[7]:
            dpg.bind_item_theme(l[0], themes[5])
        else:
            print(l[0])
            dpg.bind_item_theme(l[0], themes[l[5]])
    
def redraw(all=0):
    if all:
        rec = db.get_records('lines')
    else:
        rec = db.get_records_where('lines','forredraw=1')
    
    ids = []
    for l in rec:
        dpg.delete_item(f'{l[0]}')
        dpg.add_line_series([l[1],l[3]], [l[2], l[4]], parent=Y_AXIS_TAG,tag=f'{l[0]}') 
        ids.append(l[0])
        if l[7]:
            dpg.bind_item_theme(dpg.last_item(), themes[5])
        else:
            dpg.bind_item_theme(dpg.last_item(), themes[l[5]])
    
    db.update_multiple_records('lines','forredraw',ids,0)


def set_color():
    
    if dpg.get_value('color_1'):                 
        db.set_color('lines',0)
    elif dpg.get_value('color_2'):
        db.set_color('lines',1)
    elif dpg.get_value('color_3'):
        db.set_color('lines',2)
    elif dpg.get_value('color_4'):
        db.set_color('lines',3)
    elif dpg.get_value('color_5'):
        db.set_color('lines',4) 


    redraw()
def delete_l():
    ids = db.get_id_by_field('lines','isactive',1)
    tags = db.get_parent_by_field_unique('lines','isactive',1)
    for t in tags:
        dpg.delete_item(t)
    for i in ids:
        
        dpg.delete_item(f'{i}')
    db.delete_active('lines')
    redraw(1)


def split_l():
    for t in db.get_parent_by_field_unique('lines','isactive',1):
        dpg.delete_item(t)     

        lines_for_split = []
        ids = []
        for i,o in enumerate(db.get_records_where('lines',f"parent='{t}'")):   
            lines_for_split.append([(o[1],o[2]),(o[3],o[4])])
            ids.append(o[0])
        sett = {i for i in range(len(lines_for_split))}
    

        v = 0
        while sett:
            i = next(iter(sett))

            l = find_closest_lines(lines_for_split,lines_for_split[i][0],sett)
            dpg.add_button(label=t + f'__{v}',parent='butonss',tag=t + f'__{v}',callback=active_but)
            nice_ids = []
            for h in l:
                nice_ids.append(ids[h]) 
                sett.remove(h)
            db.update_multiple_records('lines','parent',nice_ids,f"'{t}__{v}'")
            db.update_multiple_records('lines','forredraw',nice_ids,1)  
            db.update_multiple_records('lines','isactive',nice_ids,0)
            v+=1

    redraw()

def optimize_():
    
    return
    #optimize.create_continuous_lines('temp.dxf',lines )
    
def normal_():
    normalize_lines()

def rotate_x():
    invers_lines()

def normalize_lines():
    rec = db.get_records_where('lines','isactive=1')
    xx = [r[1] for r in rec]
    xx+= [r[3] for r in rec]
    yy = [r[2] for r in rec]
    yy+= [r[4] for r in rec]
    db.increment_field_value_with_condition('lines','sx','ex','sy','ey',-min(xx),-min(yy),'isactive',1)
    redraw()

def invers_lines():
    rec = db.get_records_where('lines','isactive=1')
    xx = [r[1] for r in rec]
    xx+= [r[3] for r in rec]
    yy = [r[2] for r in rec]
    yy+= [r[4] for r in rec]

    db.inverse_field_value_with_condition('lines','sy','ey',max(yy) + min(yy),'isactive',1)
    redraw()
    


def pr(selected_files):
    global esyedaflag
    current_file = selected_files[0]
    if dpg.get_value('eraseold'):
        for t in db.get_unique_values('lines','parent'):
            dpg.delete_item(t)
        db.clear_table('lines')
        dpg.delete_item(Y_AXIS_TAG, children_only=True, slot=1)
    if '.dxf' in current_file: 
        

        if esyedaflag:
            esyedaflag = False
            normicks = ['TopLayer','BoardOutLine','Multi-Layer']
            doc = ezdxf.readfile(current_file)
            layers = doc.layers
            ll = []
            ll.append(current_file)
            dpg.add_radio_button(parent="modal_id",items=['full','border'],tag='varradio',horizontal=True,default_value='full')
            for layer in layers:
                
                ll.append(layer.dxf.name)
                if layer.dxf.name in normicks:
                    dpg.add_checkbox(label=layer.dxf.name,parent="modal_id",tag=layer.dxf.name,default_value=True)
                else:
                    dpg.add_checkbox(label=layer.dxf.name,parent="modal_id",tag=layer.dxf.name)
            dpg.add_group(horizontal=True,tag='hor_grouph',parent="modal_id")
            
            dpg.add_button(label="OK", width=75, parent='hor_grouph',callback=read_dxf_lines_from_esyeda,user_data=ll,tag='OK')
            dpg.add_button(label="Cancel", width=75, parent='hor_grouph', callback=lambda: dpg.configure_item("modal_id", show=False),tag='CANCEL')
            dpg.configure_item("modal_id", show=True)
            #dpg.configure_item("modal_id", modal=True)
            
            #lines = read_dxf_lines_from_esyeda(current_file)
            #db.add_multiple_records('lines',lines) 
        else:
            lines = read_dxf_lines(current_file)
            db.add_multiple_records('lines',lines) 
        
            redraw()
    elif '.png' in current_file:   
        lines = extract_black_lines(current_file,0.1)
        db.add_multiple_records('lines',lines)
        redraw()
    

    
###########################################
##########################################
#############################################
def test_callback():
    recolor()
####################################################
####################################################
####################################################

def esye():
    global esyedaflag
    fd.show_file_dialog()
    esyedaflag = True


dpg.create_context()

X_AXIS_TAG = "x_axis_tag"
Y_AXIS_TAG = "y_axis_tag"

current_file = None
themes = []
components = []

esyedaflag = False

db = SQLiteDatabase('example.db')
db.drop_table('lines')
db.create_table('lines', [('sx', 'REAL'), ('sy', 'REAL'), ('ex', 'REAL'), ('ey', 'REAL'),('color', 'INTEGER'), ('parent', 'TEXT'),('isactive', 'INTEGER'),('forredraw', 'INTEGER')])

with dpg.window(label="Delete Files", show=False, tag="modal_id", no_title_bar=True):
    dpg.add_text("Layers")
    dpg.add_separator()
    

with dpg.window(label="Border EsyEDA", show=False, tag="border_from_esyeda", no_title_bar=True,pos=(400,100)):
    dpg.add_text("EsyEDA border")
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text("line width ")      
        dpg.add_input_text(width=50,scientific=True,tag='border_line_width',default_value='0.1')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("count lines")      
        dpg.add_input_text(width=50,scientific=True,tag='border_line_count',default_value='10') 
    with dpg.group(horizontal=True):
        dpg.add_spacer(width=50)
        dpg.add_button(label='Apply',callback=lambda:dpg.configure_item("border_from_esyeda", show=False))
        dpg.add_spacer(width=50)

with dpg.theme(tag="coloured_line_theme1") as coloured_line_theme1:
    with dpg.theme_component():
        coloured_line_component1 = dpg.add_theme_color(dpg.mvPlotCol_Line, (0, 191, 255, 255), category=dpg.mvThemeCat_Plots)
with dpg.theme(tag="coloured_line_theme2") as coloured_line_theme2:
    with dpg.theme_component():
        coloured_line_component2 = dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 20, 147, 255), category=dpg.mvThemeCat_Plots)
with dpg.theme(tag="coloured_line_theme3") as coloured_line_theme3:
    with dpg.theme_component():
        coloured_line_component3 = dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 215, 0, 255), category=dpg.mvThemeCat_Plots)
with dpg.theme(tag="coloured_line_theme4") as coloured_line_theme4:
    with dpg.theme_component():
        coloured_line_component4 = dpg.add_theme_color(dpg.mvPlotCol_Line, (0, 255, 127, 255), category=dpg.mvThemeCat_Plots)

with dpg.theme(tag="coloured_line_theme5") as coloured_line_theme5:
    with dpg.theme_component():
        coloured_line_component5 = dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 69, 0, 255 ), category=dpg.mvThemeCat_Plots)
with dpg.theme(tag="coloured_line_theme6") as coloured_line_theme6:
    with dpg.theme_component():
        coloured_line_component6 = dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 255, 255, 255 ), category=dpg.mvThemeCat_Plots)

themes.append(coloured_line_theme1)
components.append(coloured_line_component1)
themes.append(coloured_line_theme2)
components.append(coloured_line_component2)
themes.append(coloured_line_theme3)
components.append(coloured_line_component3)
themes.append(coloured_line_theme4)
components.append(coloured_line_component4)
themes.append(coloured_line_theme5)
components.append(coloured_line_component5)
themes.append(coloured_line_theme6)
components.append(coloured_line_component6)




with dpg.theme() as plot_theme:
    with dpg.theme_component(dpg.mvPlot):
        
        dpg.add_theme_color(dpg.mvPlotCol_Line,  [80, 80, 80])## линии на поле, цифры на осях, подпись графика и оси
        
        dpg.add_theme_color(dpg.mvPlotCol_MarkerOutline,  [255, 255, 255])## цвет поля
        
        dpg.add_theme_color(dpg.mvPlotCol_FrameBg,  [0, 0, 0])## граница поля
        
        dpg.add_theme_color(dpg.mvPlotCol_PlotBorder,  [200, 200, 200])## вокруг поля (под цифрами, названиями осей)
        

with dpg.theme() as coloured_Core_theme1:
    with dpg.theme_component():
        coloured_core_component1 = dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (0, 191, 255, 255), category=dpg.mvThemeCat_Core)
        coloured_core_component11= dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 191, 255, 255), category=dpg.mvThemeCat_Core)
with dpg.theme() as coloured_Core_theme2:
    with dpg.theme_component():
        coloured_core_component2 = dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (255, 20, 147, 255), category=dpg.mvThemeCat_Core)
        coloured_core_component21= dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 20, 147, 255), category=dpg.mvThemeCat_Core)
with dpg.theme() as coloured_Core_theme3:
    with dpg.theme_component():
        coloured_core_component3 = dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (255, 215, 0, 255), category=dpg.mvThemeCat_Core)
        coloured_core_component31= dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 215, 0, 255), category=dpg.mvThemeCat_Core)
with dpg.theme() as coloured_Core_theme4 :
    with dpg.theme_component():
        coloured_core_component4 = dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (0, 255, 127, 255), category=dpg.mvThemeCat_Core)
        coloured_core_component41= dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 255, 127, 255), category=dpg.mvThemeCat_Core)
with dpg.theme() as coloured_Core_theme5:
    with dpg.theme_component():
        coloured_core_component5 = dpg.add_theme_color(dpg.mvThemeCol_CheckMark, (255, 69, 0, 255 ), category=dpg.mvThemeCat_Core)
        coloured_core_component51= dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 69, 0, 255 ), category=dpg.mvThemeCat_Core)




with dpg.theme() as enabled_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 255, 255), category=dpg.mvThemeCat_Core)


with dpg.theme() as disabled_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 0, 0), category=dpg.mvThemeCat_Core)

fd = FileDialog(callback=pr,width=800,height=400,filter_list=[".dxf",".png"])
#fd_esyeda = FileDialog(callback=esy_eda)

with dpg.viewport_menu_bar():
    with dpg.menu(label="File"):
        dpg.add_menu_item(label="Open", callback=fd.show_file_dialog)
        dpg.add_menu_item(label="Open DXF from EsyEDA", callback=esye)
        dpg.add_menu_item(label="Save As Gcode", callback=save_as_gcode)
        dpg.add_menu_item(label="Save As DXF", callback=save_as_dxf)
        
        
        with dpg.menu(label="Settings"):
            dpg.add_menu_item(label="Setting 1", callback=print_me, check=True)
            dpg.add_menu_item(label="Border from EsyEDA", callback=lambda:dpg.configure_item("border_from_esyeda", show=True))
    with dpg.menu(label="Functions"):
        dpg.add_menu_item(label="Split", callback=split_l)
        dpg.add_menu_item(label="Normalize", callback=normal_)
        dpg.add_menu_item(label="Rotate X", callback=rotate_x)
        dpg.add_menu_item(label="Delete", callback=delete_l)
        dpg.add_menu_item(label="Set Color", callback=set_color)
        dpg.add_menu_item(label="test", callback=test_callback)

    with dpg.menu(label="Widget Items"):
        dpg.add_checkbox(label="Pick Me", callback=print_me)
        dpg.add_button(label="Press Me", callback=print_me)
        dpg.add_color_picker(label="Color Me", callback=print_me)      


with dpg.window(pos=(0,0),width=900, height=725,tag='papa'):
    
    with dpg.group(horizontal=True):
        with dpg.group():
            with dpg.file_dialog(directory_selector=False, show=False, callback=save_dxf, id="file_dialog_id1", width=700 ,height=400):
                    dpg.add_file_extension(".dxf", color=(255, 0, 255, 255), custom_text="[DXF]")
            with dpg.file_dialog(directory_selector=False, show=False, callback=callback_to_gcode, id="file_dialog_id2", width=700 ,height=400):
                    dpg.add_file_extension(".gcode", color=(255, 0, 255, 255), custom_text="[GCODE]")

            
            with dpg.plot(label="DXF Plot", width=600, height=600, tag="plot",no_menus=True, no_box_select=True) as plot:
                dpg.add_plot_axis(dpg.mvXAxis, label="X-Axis", tag=X_AXIS_TAG)
                
            
                yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Y-Axis", tag=Y_AXIS_TAG)
                
                dpg.set_axis_limits_constraints(Y_AXIS_TAG,-10,310)
                dpg.set_axis_limits_constraints(X_AXIS_TAG,-10,310)
            
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("order")
                    dpg.add_text("power")
                    dpg.add_text("speed")
                with dpg.group():
                    dpg.add_checkbox(label="1",tag='color_1',callback=check_callback,default_value=True)
                    dpg.bind_item_theme(dpg.last_item(), coloured_Core_theme1)
                    dpg.add_input_text(width=50,scientific=True,tag='1_value',default_value='100')
                    dpg.add_input_text(width=50,scientific=True,tag='11_value',default_value='1000')
                with dpg.group():
                    dpg.add_checkbox(label="2",tag='color_2',callback=check_callback)
                    dpg.bind_item_theme(dpg.last_item(), coloured_Core_theme2)
                    dpg.add_input_text(width=50,scientific=True,tag='2_value',default_value='100')
                    dpg.add_input_text(width=50,scientific=True,tag='21_value',default_value='1000')
                with dpg.group():
                    dpg.add_checkbox(label="3",tag='color_3',callback=check_callback)
                    dpg.bind_item_theme(dpg.last_item(), coloured_Core_theme3)
                    dpg.add_input_text(width=50,scientific=True,tag='3_value',default_value='100')
                    dpg.add_input_text(width=50,scientific=True,tag='31_value',default_value='1000')
                with dpg.group():
                    dpg.add_checkbox(label="4",tag='color_4',callback=check_callback)
                    dpg.bind_item_theme(dpg.last_item(), coloured_Core_theme4)
                    dpg.add_input_text(width=50,scientific=True,tag='4_value',default_value='100')
                    dpg.add_input_text(width=50,scientific=True,tag='41_value',default_value='1000')
                with dpg.group():
                    dpg.add_checkbox(label="5",tag='color_5',callback=check_callback)
                    dpg.bind_item_theme(dpg.last_item(), coloured_Core_theme5)
                    dpg.add_input_text(width=50,scientific=True,tag='5_value',default_value='100')
                    dpg.add_input_text(width=50,scientific=True,tag='51_value',default_value='1000')


                
                




        with dpg.group():
            dpg.add_input_text(multiline=True, label="", default_value="", tag="multiline_input", readonly=True,width=300,height=600)
            dpg.add_checkbox(label="erase old",default_value=True,tag='eraseold')
            with dpg.group(horizontal=True):
                
                dpg.add_checkbox(label="paste numbers",default_value=False,tag='add_text',callback=rasberitesb)
                dpg.add_input_text(width=50,scientific=True,tag='insert_numbers',default_value='123')
            dpg.add_checkbox(label="change order",default_value=False,tag='change_order',callback=rasberitesb)
            dpg.add_checkbox(label="move lines",default_value=False,tag='movelines',callback=rasberitesb)
    
with dpg.item_handler_registry() as registry:
    dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Right, callback=plot_mouse_click_callback)
dpg.bind_item_handler_registry(plot, registry)


dpg.add_window(pos=(900,0),width=200, height=725,tag='butonss',label='lines')
   

dpg.create_viewport(width=1115, height=785, title="GCODE IDE")
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()