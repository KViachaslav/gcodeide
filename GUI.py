import dearpygui.dearpygui as dpg
from fdialog import FileDialog
import optimize
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import dearpygui.dearpygui as dpg
import ezdxf
import numpy as np
import math
import svgwrite
import random
from PIL import Image
import os
from ezdxf import colors
import shapely
from shapely.geometry import Point, LineString, MultiPoint,Polygon,MultiPolygon,MultiLineString
from shapely.ops import unary_union
import re
from line_manager import PolylineDatabase
import serial.tools.list_ports
import serial
import time
import xml.etree.ElementTree as ET
import re
from scipy.special import comb
import dxfgrabber
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev,splrep,BSpline
from shapely.geometry import LineString, Point, GeometryCollection
from shapely.ops import split
from shapely.geometry import box
from laser import LaserController
def active_but(sender):
    state = data_base.get_polyline_where(f"big_tag='{sender}'")
    if place_in_a_circle:
        if not state[0][4]:
            dpg.set_value('activetext',dpg.get_value('activetext') + f"{sender}\n")
        else:
            dpg.set_value('activetext',dpg.get_value('activetext').replace(f"{sender}\n", ""))
    tags = [s[1] for s in state]
    data_base.update_polylines(tags,active=False if state[0][4]==1  else True)
    data_base.update_polylines(tags,color_change_flag=True)
    dpg.bind_item_theme(sender, enabled_theme if state[0][4]==1 else disabled_theme)
    recolor()
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
        for i in data_base.get_unique_politag():
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
    return liness

def remove_close_points(points, threshold=0.02):
    if not points:
        return []

    filtered_points = [points[0]]  # Добавляем первую точку в новый массив
    last_point = np.array(points[0])

    for point in points[1:]:
        current_point = np.array(point)
        distance = np.linalg.norm(current_point - last_point)  # Вычисляем расстояние до последней добавленной точки
        
        if distance >= threshold:  # Если расстояние больше порога, добавляем точку
            filtered_points.append(point)
            last_point = current_point  # Обновляем последнюю добавленную точку

    return filtered_points

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
    #находит точку пересечения между отрезком, 
    # заданным координатами двух его концов 
    # и горизонтальной линией

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

def Polygon_to_lines(union_polygon,num_lines,width_lines,nice_path):

    c = 0
    for k in range(num_lines):

        tunion_polygon = []
        if union_polygon.geom_type == 'Polygon':
            
            xm, ym = union_polygon.exterior.xy
            

            data_base.add_polyline(nice_path+f"{c}",nice_path,0, False, True, False)
            #data_base.add_coordinates(nice_path+f"{c}",[(x_,y_) for x_,y_ in zip(xm,ym)])
            data_base.add_coordinates(nice_path+f"{c}",remove_close_points([(x_,y_) for x_,y_ in zip(xm,ym)]))
            c+=1
            
            for inter in union_polygon.interiors:
                xm, ym = inter.xy
                data_base.add_polyline(nice_path+f"{c}",nice_path,0, False, True, False)
                #data_base.add_coordinates(nice_path+f"{c}",[(x_,y_) for x_,y_ in zip(xm,ym)])
                data_base.add_coordinates(nice_path+f"{c}",remove_close_points([(x_,y_) for x_,y_ in zip(xm,ym)]))
                c+=1
            
            pol = union_polygon.buffer(width_lines,quad_segs=0)
            if isinstance(pol, MultiPolygon):
                for single_polygon in pol.geoms:
                    tunion_polygon.append(single_polygon)
            else:
                tunion_polygon.append(pol)
        else:
           

            for p in union_polygon.geoms:
                
                
                
                xm, ym = p.exterior.xy
                data_base.add_polyline(nice_path+f"{c}",nice_path,0, False, True, False)
                #data_base.add_coordinates(nice_path+f"{c}",[(x_,y_) for x_,y_ in zip(xm,ym)])
                data_base.add_coordinates(nice_path+f"{c}",remove_close_points([(x_,y_) for x_,y_ in zip(xm,ym)]))
                c+=1
                for inter in p.interiors:
                    xm, ym = inter.xy
                    data_base.add_polyline(nice_path+f"{c}",nice_path,0, False, True, False)
                    #data_base.add_coordinates(nice_path+f"{c}",[(x_,y_) for x_,y_ in zip(xm,ym)])
                    data_base.add_coordinates(nice_path+f"{c}",remove_close_points([(x_,y_) for x_,y_ in zip(xm,ym)]))
                    c+=1
                
                pol = p.buffer(width_lines,quad_segs=0)
                if isinstance(pol, MultiPolygon):
                    for single_polygon in pol.geoms:
                        tunion_polygon.append(single_polygon)
                else:
                    tunion_polygon.append(pol)
        union_polygon = unary_union(MultiPolygon([p for p in tunion_polygon]))






def adjust_segments(segment1, segment2, change):
    vector1 = (segment1.coords[1][0] - segment1.coords[0][0],
               segment1.coords[1][1] - segment1.coords[0][1])
    
    length1 = (vector1[0]**2 + vector1[1]**2)**0.5
    if length1 == 0:
        raise ValueError("Segment 1 is a point, cannot shift.")
    
    unit_vector1 = (vector1[0] / length1, vector1[1] / length1)
    
    normal_vector1 = (-unit_vector1[1], unit_vector1[0]) 
    
    vector2 = (segment2.coords[1][0] - segment2.coords[0][0],
               segment2.coords[1][1] - segment2.coords[0][1])
    
    length2 = (vector2[0]**2 + vector2[1]**2)**0.5
    if length2 == 0:
        raise ValueError("Segment 2 is a point, cannot shift.")
    
    unit_vector2 = (vector2[0] / length2, vector2[1] / length2)
    
    normal_vector2 = (-unit_vector2[1], unit_vector2[0]) 
    
    half_change = change / 2  
    
    multi_point = MultiPoint([(segment1.coords[0][0] + normal_vector1[0] * half_change,
         segment1.coords[0][1] + normal_vector1[1] * half_change),
        (segment1.coords[1][0] + normal_vector1[0] * half_change,
         segment1.coords[1][1] + normal_vector1[1] * half_change),
         (segment2.coords[1][0] - normal_vector2[0] * half_change,
         segment2.coords[1][1] - normal_vector2[1] * half_change),
        (segment2.coords[0][0] - normal_vector2[0] * half_change,
         segment2.coords[0][1] - normal_vector2[1] * half_change)])
    convex_hull = multi_point.convex_hull
    return convex_hull
    
def scale_polygon_horizontal(polygon, scale_factor):
    centroid = polygon.centroid
    x, y = polygon.exterior.xy
    dx = max(x) - min(x)
    scale_factor = (scale_factor + dx)/dx
    new_coords = [(centroid.x + (xi - centroid.x) * scale_factor, yi) for xi, yi in zip(x, y)]
    return Polygon(new_coords)
def get_radius(a,b,x,y):
    a = 0.11
    b = 0.04
    if y == 0:
        return b
    if x == 0:
        return a
    else:
        ans = a*b/np.sqrt((b*np.cos(np.arctan(x/y)))**2 + (a*np.sin(np.arctan(x/y)))**2) 
        return ans + (a-ans) * 0.5

def combine_lines(lines):
    # Словари для хранения начала и конца линий
    line_dict = {}
    for i, line in enumerate(lines):
        start, end = tuple(line[0]), tuple(line[1])
        line_dict[(start, end)] = i
    
    combined_line = []
    
    current_line = lines[0]
    combined_line.extend(current_line)
    used_lines = {line_dict[tuple(current_line)]}

    while len(used_lines) < len(lines):
        last_point = current_line[1] 

        found = False
        for line in lines:
            if line_dict.get((tuple(line[0]), tuple(line[1]))) not in used_lines:
                if line[0] == last_point:
                    current_line = line
                    combined_line.append(current_line[1])  # Добавляем только конечную точку
                    used_lines.add(line_dict[tuple(line)])
                    found = True
                    break
                elif line[1] == last_point:
                    current_line = line[::-1]  # Инвертируем линию
                    combined_line.append(current_line[1])  # Добавляем только конечную точку
                    used_lines.add(line_dict[tuple(current_line)])
                    found = True
                    break
        
        if not found:
            raise ValueError("Требуемый контур не может быть собран")

    return combined_line

def read_dxf_lines_from_esyeda(sender, app_data, user_data):
    for_correct = 0.09 #0.1 было
    for_buffer = 0.08
    for_buffer2 = 0.057 #47  было
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
        for i in data_base.get_unique_politag():
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
    borderlin = []
    polygons = []
    polygons2 = []
    dyrki = []
    for circle in msp.query('CIRCLE'):
        layer = circle.dxf.layer
        
        if layer in layers:
            center = circle.dxf.center    
            num_points = 24 
            radius = circle.dxf.radius + for_buffer
            radius2 = circle.dxf.radius + for_buffer2
            polygons.append(scale_polygon_horizontal(Polygon([(center.x + radius * math.cos(2 * math.pi * i / num_points),center.y + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]),for_correct))
            polygons2.append(scale_polygon_horizontal(Polygon([(center.x + radius2 * math.cos(2 * math.pi * i / num_points),center.y + radius2 * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]),for_correct))
            dyrki.append([(center.x + 0.05 * math.cos(2 * math.pi * i / 6),center.y +  0.05 * math.sin(2 * math.pi * i / 6))for i in range(7)])

        
    for polyline in msp.query('LWPOLYLINE'):
        layer = polyline.dxf.layer
        if layer in layers:
            if dpg.get_value('widthborder0') and layer == 'BoardOutLine':
                w = 0
            else:
                w = polyline.dxf.const_width
            points = polyline.get_points()  
        
            num_points = 20
            radius = w/2 + for_buffer
            radius2 = w/2 + for_buffer2
            polygons.append(scale_polygon_horizontal(Polygon([(points[0][0] + radius * math.cos(2 * math.pi * i / num_points),points[0][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]),for_correct))    
            polygons2.append(scale_polygon_horizontal(Polygon([(points[0][0] + radius2 * math.cos(2 * math.pi * i / num_points),points[0][1] + radius2 * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]),for_correct))    
            for j in range(len(points) - 1):
                num_points = 10
                radius = w/2 + for_buffer
                boundaries = calculate_boundary_coordinates(points[j][0], points[j][1], points[j + 1][0], points[j + 1][1], w + for_buffer*2)
                p = adjust_segments(LineString([(boundaries['right_end'][0],boundaries['right_end'][1]), (boundaries['right_start'][0],boundaries['right_start'][1])]),LineString([ (boundaries['left_end'][0],boundaries['left_end'][1]),(boundaries['left_start'][0],boundaries['left_start'][1])]) ,get_radius(0.16,0.04,(points[j][0] - points[j+1][0]),(points[j][1] - points[j+1][1]))-0.04)
                
                radius2 = w/2 + for_buffer2
                boundaries2 = calculate_boundary_coordinates(points[j][0], points[j][1], points[j + 1][0], points[j + 1][1], w + for_buffer2*2)
                p2 = adjust_segments(LineString([(boundaries2['right_end'][0],boundaries2['right_end'][1]), (boundaries2['right_start'][0],boundaries2['right_start'][1])]),LineString([ (boundaries2['left_end'][0],boundaries2['left_end'][1]),(boundaries2['left_start'][0],boundaries2['left_start'][1])]) ,get_radius(0.16,0.04,(points[j][0] - points[j+1][0]),(points[j][1] - points[j+1][1]))-0.04)
                
                polygons.append(scale_polygon_horizontal(Polygon([(points[j + 1][0] + radius * math.cos(2 * math.pi * i / num_points),points[j + 1][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]),for_correct))
                polygons.append(p)
                #polygons.append(Polygon([(boundaries['left_start'][0],boundaries['left_start'][1]), (boundaries['left_end'][0],boundaries['left_end'][1]), (boundaries['right_end'][0],boundaries['right_end'][1]), (boundaries['right_start'][0],boundaries['right_start'][1])]))
                polygons2.append(scale_polygon_horizontal(Polygon([(points[j + 1][0] + radius2 * math.cos(2 * math.pi * i / num_points),points[j + 1][1] + radius2 * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]),for_correct))
                if p2.geom_type == "Polygon":
                    polygons2.append(p2)
                #polygons2.append(Polygon([(boundaries2['left_start'][0],boundaries2['left_start'][1]), (boundaries2['left_end'][0],boundaries2['left_end'][1]), (boundaries2['right_end'][0],boundaries2['right_end'][1]), (boundaries2['right_start'][0],boundaries2['right_start'][1])]))

        if layer == 'BoardOutLine' and full:
            w = polyline.dxf.const_width
            points = polyline.get_points()
            borderlin.append(LineString([(points[i][0],points[i][1])for i in range(len(points))]))

            num_points = 10
            radius = w/2 + for_buffer
            border.append(Polygon([(points[0][0] + radius * math.cos(2 * math.pi * i / num_points),points[0][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))  
            for j in range(len(points) - 1):
                num_points = 20
                radius = w/2 + for_buffer
                boundaries = calculate_boundary_coordinates(points[j][0], points[j][1], points[j + 1][0], points[j + 1][1], w + for_buffer*2)
                border.append(Polygon([(points[j + 1][0] + radius * math.cos(2 * math.pi * i / num_points),points[j + 1][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))
                
                border.append(Polygon([(boundaries['left_start'][0],boundaries['left_start'][1]), (boundaries['left_end'][0],boundaries['left_end'][1]), (boundaries['right_end'][0],boundaries['right_end'][1]), (boundaries['right_start'][0],boundaries['right_start'][1])]))





    for hatch in msp.query('HATCH'):
        layer = hatch.dxf.layer
        if layer in layers:
            for path in hatch.paths:
                points = path.vertices
                if len(points) > 2:
                    polygons.append(scale_polygon_horizontal(Polygon([(points[i][0],points[i][1]) for i in range(len(points))]).buffer(for_buffer,quad_segs=2),for_correct))
                    polygons2.append(scale_polygon_horizontal(Polygon([(points[i][0],points[i][1]) for i in range(len(points))]).buffer(for_buffer2,quad_segs=2),for_correct))    
    lins = []
    if full:
        ex = shapely.envelope(unary_union(MultiPolygon([p for p in border])))
        
        if ex.geom_type == "Polygon":
            xm, ym = ex.exterior.xy
            xmin = min(xm)
            xmax = max(xm)
            ymin = min(ym)
            ymax = max(ym)
            lins = MultiLineString([((xmin, y), (xmax, y))for y in np.arange(ymin,ymax,width_lines)])
            
            combined_line = shapely.line_merge(MultiLineString(borderlin))
            
            if combined_line.geom_type == "LineString":
                polygon = Polygon(combined_line.coords).buffer(1,quad_segs=2)
            else:
                # polygon = Polygon(combined_line.geoms[1].coords).buffer(1,quad_segs=2)

                pps = [Polygon(combined_line.geoms[i].coords) for i in range(len(combined_line.geoms))]
                fl = False
                for i in range(len(pps)):
                    for j in range(len(pps)):
                        if shapely.contains_properly(pps[i],pps[j]):
                            ans = pps[i]
                            for k in range(len(pps)):
                                if k != i:
                                    ans = ans.difference(pps[k])      
                            polygon = ans.buffer(1,quad_segs=2)  
                            
                            fl = True
                            break 
                    if fl:
                        break
                
            intersection = lins.intersection(polygon)
            linn = intersection.difference(unary_union(MultiPolygon([p for p in polygons])))
            Polygon_to_lines(unary_union(MultiPolygon([p for p in polygons2])),1,width_lines,nice_path+ '_border')
            redraw()
            c = 0
            for l in dyrki:
                
                data_base.add_polyline(nice_path+f"__{c}" ,nice_path,0, False, True, False)
                data_base.add_coordinates(nice_path+f"__{c}",l)
                c+=1
                redraw()

            for l in linn.geoms:
                coords = []
                coords.append((round(l.coords[0][0],4),  round(l.coords[0][1],4)))
                coords.append((round(l.coords[1][0],4), round(l.coords[1][1],4)))           
            
                data_base.add_polyline(nice_path+f"{c}" ,nice_path,0, False, True, False)
                data_base.add_coordinates(nice_path+f"{c}",coords)
                c+=1
                redraw()
            
            
            
            dpg.add_button(label=nice_path + '_border',parent='butonss',tag=nice_path + '_border',callback=active_but)
            
    else:
        Polygon_to_lines(unary_union(MultiPolygon([p for p in polygons2])),num_lines,width_lines,nice_path)
        
    redraw()


def read_dxf_lines_from_esyeda2(sender, app_data, user_data):

    for_buffer = 0.08
    for_buffer2 = 0.05
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
        for i in data_base.get_unique_politag():
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
    polygons2 = []
    for circle in msp.query('CIRCLE'):
        layer = circle.dxf.layer
        
        if layer in layers:
            center = circle.dxf.center    
            num_points = 24 
            radius = circle.dxf.radius + for_buffer
            radius2 = circle.dxf.radius + for_buffer2
            polygons.append(Polygon([(center.x + radius * math.cos(2 * math.pi * i / num_points),center.y + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))
            polygons2.append(Polygon([(center.x + radius2 * math.cos(2 * math.pi * i / num_points),center.y + radius2 * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))

           
        
    for polyline in msp.query('LWPOLYLINE'):
        layer = polyline.dxf.layer
        if layer in layers:
            w = polyline.dxf.const_width
            points = polyline.get_points()  
        
            num_points = 20
            radius = w/2 + for_buffer
            radius2 = w/2 + for_buffer2
            polygons.append(Polygon([(points[0][0] + radius * math.cos(2 * math.pi * i / num_points),points[0][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))    
            polygons2.append(Polygon([(points[0][0] + radius2 * math.cos(2 * math.pi * i / num_points),points[0][1] + radius2 * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))    
            for j in range(len(points) - 1):
                num_points = 10
                radius = w/2 + for_buffer
                boundaries = calculate_boundary_coordinates(points[j][0], points[j][1], points[j + 1][0], points[j + 1][1], w + for_buffer*2)
                radius2 = w/2 + for_buffer2
                boundaries2 = calculate_boundary_coordinates(points[j][0], points[j][1], points[j + 1][0], points[j + 1][1], w + for_buffer2*2)
                polygons.append(Polygon([(points[j + 1][0] + radius * math.cos(2 * math.pi * i / num_points),points[j + 1][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))
                
                polygons.append(Polygon([(boundaries['left_start'][0],boundaries['left_start'][1]), (boundaries['left_end'][0],boundaries['left_end'][1]), (boundaries['right_end'][0],boundaries['right_end'][1]), (boundaries['right_start'][0],boundaries['right_start'][1])]))
                polygons2.append(Polygon([(points[j + 1][0] + radius2 * math.cos(2 * math.pi * i / num_points),points[j + 1][1] + radius2 * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))
                
                polygons2.append(Polygon([(boundaries2['left_start'][0],boundaries2['left_start'][1]), (boundaries2['left_end'][0],boundaries2['left_end'][1]), (boundaries2['right_end'][0],boundaries2['right_end'][1]), (boundaries2['right_start'][0],boundaries2['right_start'][1])]))

        if layer == 'BoardOutLine' and full:
            w = polyline.dxf.const_width
            points = polyline.get_points()
            num_points = 10
            radius = w/2 + for_buffer
            border.append(Polygon([(points[0][0] + radius * math.cos(2 * math.pi * i / num_points),points[0][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))  
            for j in range(len(points) - 1):
                num_points = 20
                radius = w/2 + for_buffer
                boundaries = calculate_boundary_coordinates(points[j][0], points[j][1], points[j + 1][0], points[j + 1][1], w + for_buffer*2)
                border.append(Polygon([(points[j + 1][0] + radius * math.cos(2 * math.pi * i / num_points),points[j + 1][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))
                
                border.append(Polygon([(boundaries['left_start'][0],boundaries['left_start'][1]), (boundaries['left_end'][0],boundaries['left_end'][1]), (boundaries['right_end'][0],boundaries['right_end'][1]), (boundaries['right_start'][0],boundaries['right_start'][1])]))





    for hatch in msp.query('HATCH'):
        layer = hatch.dxf.layer
        if layer in layers:
            for path in hatch.paths:
                points = path.vertices
                if len(points) > 2:
                    polygons.append(Polygon([(points[i][0],points[i][1]) for i in range(len(points))]).buffer(for_buffer,quad_segs=2))
                    polygons2.append(Polygon([(points[i][0],points[i][1]) for i in range(len(points))]).buffer(for_buffer2,quad_segs=2))    
    lins = []
    if full:
        ex = shapely.envelope(unary_union(MultiPolygon([p for p in border])))
        
        if ex.geom_type == "Polygon":
            xm, ym = ex.exterior.xy
            xmin = min(xm)
            xmax = max(xm)
            ymin = min(ym)
            ymax = max(ym)
            lins = MultiLineString([((xmin, y), (xmax, y))for y in np.arange(ymin,ymax,width_lines)])

            linn = lins.difference(unary_union(MultiPolygon([p for p in polygons])))
            
            c = 0
            for l in linn.geoms:
                coords = []
                coords.append((round(l.coords[0][0],4),  round(l.coords[0][1],4)))
                coords.append((round(l.coords[1][0],4), round(l.coords[1][1],4)))           
            
                data_base.add_polyline(nice_path+f"{c}" ,nice_path,0, False, True, False)
                data_base.add_coordinates(nice_path+f"{c}",coords)
                c+=1
                redraw()
            Polygon_to_lines(unary_union(MultiPolygon([p for p in polygons2])),1,width_lines,nice_path+ '_border')
            
            
            dpg.add_button(label=nice_path + '_border',parent='butonss',tag=nice_path + '_border',callback=active_but)
            
    else:
        Polygon_to_lines(unary_union(MultiPolygon([p for p in polygons2])),num_lines,width_lines,nice_path)
        
    redraw()


def read_dxf_with_grabber(file_path):
    nice_path = find_nice_path(os.path.basename(file_path))
    global borderflag
    dpg.add_button(label=nice_path,parent='butonss',tag=nice_path,callback=active_but)
    dxf = dxfgrabber.readfile(file_path)
    
    ll = []
    lll = {}
    pattern = r'^power(\d+)speed(\d+)$'
    h = 1
    for layer in dxf.layers:
        match = re.match(pattern, layer.name)
        if match:
            ll.append(layer.name)
            power = int(match.group(1))
            speed = int(match.group(2))
            dpg.set_value(f"{h}_value",power)
            dpg.set_value(f"{h}1_value",speed)
            lll[layer.name] = h - 1 
            h+=1
    counter = 0
    for entity in dxf.entities:
        if borderflag and entity.layer == 'BoardOutLine' or not borderflag:
            if entity.dxftype == 'LINE':
                data_base.add_polyline(nice_path+f"_line_"+f"{counter}",nice_path,0, False, True, False)
                data_base.add_coordinates(nice_path+f"_line_"+f"{counter}", [[entity.start[0],entity.start[1]], [entity.end[0],entity.end[1]]])
            elif entity.dxftype == 'CIRCLE':
                
                center = entity.center 
                radius = entity.radius
                num_points = 50  
                layer = entity.layer
                points = [
                    (
                        center[0] + radius * math.cos(2 * math.pi * i / num_points),
                        center[1] + radius * math.sin(2 * math.pi * i / num_points)
                    )
                    for i in list(range(num_points)) + [0]
                ]
                if layer in ll:
                    data_base.add_polyline(nice_path+f"_circle_"+f"{counter}",nice_path,lll[layer], False, True, False)
                else:
                    data_base.add_polyline(nice_path+f"_circle_"+f"{counter}",nice_path,0, False, True, False)
                data_base.add_coordinates(nice_path+f"_circle_"+f"{counter}", points)
                counter+=1
            elif entity.dxftype == 'SPLINE':
                degree = entity.degree
                if hasattr(entity, 'knots') and entity.knots:  
                    knot_vector = np.array(entity.knots)
                    control_points = np.array(entity.control_points)
                    
                    x = control_points[:, 0]
                    y = control_points[:, 1]   
                    weights = getattr(entity, 'weights', None) 
                    if weights is not None and not all(w == 1.0 for w in weights):
                        print(f"Предупреждение: Сплайн является NURBS. Используется нерациональное приближение.") 
                    try:   
                        u_min = knot_vector[degree]
                        u_max = knot_vector[len(knot_vector) - degree - 1]
                        u_range = np.linspace(u_min, u_max, 20)
                        
                        spline_x = BSpline(knot_vector, x, degree)
                        spline_y = BSpline(knot_vector, y, degree)
                        
                        curve_x = spline_x(u_range)
                        curve_y = spline_y(u_range)
                        
                        # ax.plot(curve_x, curve_y, '-', label=f'Сплайн {spline_count} (BSpline)')
                        data_base.add_polyline(nice_path+f"_spline_"+f"{counter}",nice_path,0, False, True, False)
                        data_base.add_coordinates(nice_path+f"_spline_"+f"{counter}", [(x,y)for x,y, in zip(curve_x,curve_y)])
            
                        
                    except Exception as e:
                        print(f"Ошибка SciPy при обработке BSpline : {e}")

                elif hasattr(entity, 'fit_points') and entity.fit_points:
                    
                    fit_points = np.array(entity.fit_points)
                    x = fit_points[:, 0]
                    y = fit_points[:, 1]
                    
                    try:
                    
                        tck, u = splprep([x, y], k=degree, s=0) 
                        u_new = np.linspace(u.min(), u.max(), 20)
                        curve_points = splev(u_new, tck)
                        
                        
                        data_base.add_polyline(nice_path+f"_spline_"+f"{counter}",nice_path,0, False, True, False)
                        data_base.add_coordinates(nice_path+f"_spline_"+f"{counter}", [(x,y)for x,y, in zip(curve_points[0],curve_points[1])])
            
                    except Exception as e:
                        print(f"Ошибка SciPy при интерполяции (Fit Points) : {e}")

                else:
                    print(f"Сплайн пропущен: Нет ни knots, ни fit_points.")


            elif entity.dxftype == 'ARC':
                
                center = entity.center 
                radius = entity.radius  
                start_angle = entity.start_angle
                end_angle = entity.end_angle
                layer = entity.layer
                if radius<10:
                    points = arc_to_lines(center, radius, start_angle, end_angle,10)
                else:
                    points = arc_to_lines(center, radius, start_angle, end_angle,50)
                if layer in ll:
                    data_base.add_polyline(nice_path+f"_arc_"+f"{counter}",nice_path,lll[layer], False, True, False)
                else:
                    data_base.add_polyline(nice_path+f"_arc_"+f"{counter}",nice_path,0, False, True, False)
                data_base.add_coordinates(nice_path+f"_arc_"+f"{counter}", points)
                counter+=1

            elif entity.dxftype == 'LWPOLYLINE':
            
                layer = entity.layer
                points = entity.points 
                coords = []
                for i in range(len(points)):
                    coords.append((round(points[i][0],4),  round(points[i][1],4)))
                if entity.is_closed:
                    coords.append((round(points[0][0],4),  round(points[0][1],4)))
                if layer in ll:
                    data_base.add_polyline(nice_path+f"_lwpoly_"+f"{counter}",nice_path,lll[layer], False, True, False)
                else:
                    data_base.add_polyline(nice_path+f"_lwpoly_"+f"{counter}",nice_path,0, False, True, False)
                data_base.add_coordinates(nice_path+f"_lwpoly_"+f"{counter}", coords)

            elif entity.dxftype == 'POLYLINE':
                
                layer = entity.layer
                points = entity.points 
                coords = []
                for i in range(len(points)):
                    coords.append((round(points[i][0],4),  round(points[i][1],4)))
                if entity.is_closed:
                    coords.append((round(points[0][0],4),  round(points[0][1],4)))
                if layer in ll:
                    data_base.add_polyline(nice_path+f"_lwpoly_"+f"{counter}",nice_path,lll[layer], False, True, False)
                else:
                    data_base.add_polyline(nice_path+f"_lwpoly_"+f"{counter}",nice_path,0, False, True, False)
                data_base.add_coordinates(nice_path+f"_lwpoly_"+f"{counter}", coords)
            elif entity.dxftype == 'HATCH':
                print(f"Pattern Type: {entity.pattern_type}, Loop Count: {len(entity.loops)}")
                print(f"Type: {entity.dxftype}, Layer: {entity.layer}")
            
            elif entity.dxftype == '3DFACE':
                print(f"Vertices: {entity.points}")
                print(f"Type: {entity.dxftype}, Layer: {entity.layer}")
            else:
                print(f"{entity.dxftype} sss")
            counter+=1
           
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
    mods = []
    Nums = set(nums)
    lins = []
    closest_point,I,mode,min_distance = find_closest_pointt(lines, target_point,Nums)
    current_point = closest_point
    while 1:
        closest_point,I,mode,min_distance = find_closest_pointt(lines, current_point,Nums)

        if min_distance < 0.1:
            Nums.remove(I)
            lins.append(I)
            mods.append(mode)
            if mode:
                current_point = lines[I][1]
            else:
                current_point = lines[I][0]
        else:
            return lins,mods

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
def get_gcode_from_tags(tags):
    gcode_lines = []
    h = 1
    curr_pos = [0,0]
    for tag in tags:
        tag_lines = ['G90','G21','M4 S0']
        power = dpg.get_value(f"{h}_value")
        speed = dpg.get_value(f"{h}1_value")
        repeat = int(dpg.get_value(f"{h}11_value"))
        sett = set(tag)
        
        i = 0 
        while sett:
            
            poly = tag[i]
            i+=1
            sett.remove(poly)
            coords = data_base.get_coordinates(poly)
            if len(coords) > 1:
                
                tag_lines.append(f"G0 X{round(coords[0][0],4)} Y{round(coords[0][1],4)}")
                tag_lines.append(f"F{speed}")
                tag_lines.append(f"S{power}")
                tag_lines.append(f"G1 X{round(coords[1][0],4)} Y{round(coords[1][1],4)}")
                if len(coords) > 2:
                    for coord in coords[2:]:
                        tag_lines.append(f"X{round(coord[0],4)} Y{round(coord[1],4)}")
                tag_lines.append("S0")

        for i in range(repeat):
            
            for j in tag_lines:
                
                gcode_lines.append(j)

    gcode_lines.append(f"M5 S0")
    return gcode_lines

def callback_to_gcode2(sender, app_data, user_data):
    current_file = app_data['file_path_name']
    
    tag0 = data_base.get_tag_where('color=0')
    tag1 = data_base.get_tag_where('color=1')
    tag2 = data_base.get_tag_where('color=2')
    tag3 = data_base.get_tag_where('color=3')
    tag4 = data_base.get_tag_where('color=4')

    gcode_lines = get_gcode_from_tags([tag0,tag1,tag2,tag3,tag4])
    with open(current_file, 'w') as f:
        f.write("\n".join(gcode_lines))

    dpg.set_value('multiline_input',"\n".join(gcode_lines))



def save_as_dxf():
    dpg.show_item("file_dialog_id1")
def save_sel_as_dxf():
    dpg.show_item("file_dialog_id11")
def save_dxf(sender, app_data, user_data):
    current_file = app_data['file_path_name']
    doc = ezdxf.new()
    msp = doc.modelspace()

    tag0 = data_base.get_tag_where('color=0')
    tag1 = data_base.get_tag_where('color=1')
    tag2 = data_base.get_tag_where('color=2')
    tag3 = data_base.get_tag_where('color=3')
    tag4 = data_base.get_tag_where('color=4')
    tags = [tag0,tag1,tag2,tag3,tag4]
    h = 1
    
    for tag in tags:
        power = dpg.get_value(f"{h}_value")
        speed = dpg.get_value(f"{h}1_value")
        layer_name = f"power{power}speed{speed}"
        doc.layers.new(name=layer_name, dxfattribs={'color': 7})  # 7 - цвет белый
        h+=1
        for t in tag:
            coords = data_base.get_coordinates(t)
            
            msp.add_lwpolyline(coords, close=False,dxfattribs={'layer': layer_name})
    doc.saveas(current_file)
def save_sel_dxf(sender, app_data, user_data):
    current_file = app_data['file_path_name']
    doc = ezdxf.new()
    msp = doc.modelspace()

    tag0 = data_base.get_tag_where('color=0 AND active=True')
    tag1 = data_base.get_tag_where('color=1 AND active=True')
    tag2 = data_base.get_tag_where('color=2 AND active=True')
    tag3 = data_base.get_tag_where('color=3 AND active=True')
    tag4 = data_base.get_tag_where('color=4 AND active=True')
    tags = [tag0,tag1,tag2,tag3,tag4]
    h = 1
    
    for tag in tags:
        power = dpg.get_value(f"{h}_value")
        speed = dpg.get_value(f"{h}1_value")
        layer_name = f"power{power}speed{speed}"
        doc.layers.new(name=layer_name, dxfattribs={'color': 7})  # 7 - цвет белый
        h+=1
        for t in tag:
            coords = data_base.get_coordinates(t)
            
            msp.add_lwpolyline(coords, close=False,dxfattribs={'layer': layer_name})
    doc.saveas(current_file)

def check_callback(sender):
    for i in ['color_1','color_2','color_3','color_4','color_5']:
         if i != sender:
              dpg.set_value(i,False)

def print_me(sender):
    print(f"Menu Item: {sender}")


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
        
        for s in ['change_order','add_text','movelines','select','rotate','center','split']:
            if sender != s:
                dpg.set_value(s,False)

def calback_reselect(sender):
    
    if dpg.get_item_label("selectall") == 'select all':
        data_base.select_all()
        dpg.set_item_label("selectall", "hide all")
    else:
        data_base.hide_all()
        dpg.set_item_label("selectall", "select all")
    
    recolor()

def rotate_points(points, center, angle_degrees):
    
    angle_radians = np.deg2rad(angle_degrees)
    
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)
    
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    center = np.array(center)
    points_shifted = points - center
    
    points_rotated_shifted = points_shifted @ rotation_matrix.T
    
    points_final = points_rotated_shifted + center
    
    return points_final

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
    
    # Функция инволюты (inv_t)
    def inv_t(t):
        return np.tan(t) - t

    # Угол профиля на вершине зуба (t_a)
    try:
        t_a = math.acos(r_b / r_a)
    except ValueError:
        t_a = 0 # Фактически ошибка, но для кода пусть будет 0

    results = {
        "r": r_delitelni, "z": z_zubiev, "alpha_rad": alpha_rad, 
        "m": m, "d": d, "r_b": r_b, "r_a": r_a, "r_f": r_f,
        "h_a": h_a, "h_f": h_f, "t_a": t_a, "inv_t": inv_t
    }
    return results

def plot_gear_profile(gear_data, num_points=100, num_teeth_to_show=2):
    
    r, z = gear_data['r'], gear_data['z']
    r_b, r_a, r_f = gear_data['r_b'], gear_data['r_a'], gear_data['r_f']
    t_a = gear_data['t_a']
    inv_t = gear_data['inv_t']
    
    t_values = np.linspace(0, t_a, num_points)
    
    rho = r_b / np.cos(t_values)
    phi = inv_t(t_values) + (math.pi / (2 * z))
    
    x_evo = rho * np.cos(phi)
    y_evo = rho * np.sin(phi)
    
    c, s = np.cos(-(2 * math.pi) / z/2), np.sin(-(2 * math.pi) / z/2)
    R = np.array([[c, -s], [s, c]])
    
    rotated_coords = R @ np.array([x_evo, y_evo])
    x_evo = rotated_coords[0]
    y_evo = rotated_coords[1]
    x_tooth = np.concatenate([x_evo, np.flip(x_evo)])
    y_tooth = np.concatenate([y_evo, np.flip(-y_evo)]) # Зеркальное отражение
    
    pitch_angle = (2 * math.pi) / z
    
    half_pitch_angle = pitch_angle / 2
    
    angle_shift_tooth = pitch_angle
    
    
    lines = []
    for i in range(z):
        if i >= num_teeth_to_show:
            break
            
        rotation_angle = i * angle_shift_tooth
        
        c, s = np.cos(rotation_angle), np.sin(rotation_angle)
        R = np.array([[c, -s], [s, c]])
        
        rotated_coords = R @ np.array([x_tooth, y_tooth])
        lines += list(np.transpose(rotated_coords))
        
    
    return lines
def get_sector_points(diameter, radius, num_circles, angle_range=100, points_per_sector=100):
    # Радиус окружности, на которой располагаются окружности
    center_radius = diameter / 2
    angles = np.linspace(0, 2 * np.pi, num_circles*2, endpoint=False)
    
    diss = distance([center_radius * np.cos(angles[0]),center_radius * np.sin(angles[0])],[center_radius * np.cos(angles[1]),center_radius * np.sin(angles[1])])
    diss -= radius
    sectors_points = {}
    ansx = []
    ansy = []
    angle_range2 = 2 * np.pi - angle_range
    for i, angle in enumerate(angles):
        if i%2 == 0:
            center_x = center_radius * np.cos(angle)
            center_y = center_radius * np.sin(angle)

            angle_rad_start = np.pi + angle + angle_range/2 
            angle_rad_end = np.pi + angle - angle_range / 2
        
            sector_angles = np.linspace(angle_rad_start, angle_rad_end, points_per_sector)
            
            sector_x = center_x + radius * np.cos(sector_angles)
            sector_y = center_y + radius * np.sin(sector_angles)
            ansx.extend(sector_x)
            ansy.extend(sector_y)
            sectors_points[i] = (sector_x, sector_y)
        else:
            center_x = center_radius * np.cos(angle)
            center_y = center_radius * np.sin(angle)

            angle_rad_end =  angle + angle_range2/2 
            angle_rad_start =  angle - angle_range2 / 2
        
            sector_angles = np.linspace(angle_rad_start, angle_rad_end, points_per_sector)
            
            sector_x = center_x + diss * np.cos(sector_angles)
            sector_y = center_y + diss * np.sin(sector_angles)
            ansx.extend(sector_x)
            ansy.extend(sector_y)
            sectors_points[i] = (sector_x, sector_y)

    #return sectors_points
    return [ansx,ansy]

def Epitrohoida_callback():
    dpg.configure_item("Epitrohoida_window", show=False)
    nice_path = 'Epitrohoida'
    iter = 1
    while 1:
        for i in data_base.get_unique_politag():
            if i == nice_path:
                nice_path = 'Epitrohoida' + f' ({iter})'
                iter +=1
        else:
            break
    N = int(dpg.get_value('N_'))
    n = int(dpg.get_value('n_'))
    Dzkk = float(dpg.get_value('Dzkk'))
    Dkk = float(dpg.get_value('Dkk'))
    e = float(dpg.get_value('e'))

    nnp = int(dpg.get_value('countp'))
    
    Doe = n * Dkk/N
    Doeo = Dkk - Doe 
    Roe = Doe/2
    Roeo = Doeo/2
    theta = np.linspace(0, 2 * np.pi, nnp)

    x = (Roe + Roeo) * np.cos(theta) - e * np.cos((Roe+Roeo) * theta/Roeo)
    y = (Roe + Roeo) * np.sin(theta) - e * np.sin((Roe+Roeo) * theta/Roeo)
    # data_base.add_polyline(nice_path + "_",nice_path,0, False, True, False)
    # data_base.add_coordinates(nice_path + "_", [(xx,yy) for xx,yy in zip(x,y)])
    dpg.add_button(label=nice_path  ,parent='butonss',tag=nice_path ,callback=active_but)

    fp = [(xx,yy) for xx,yy in zip(x,y)]
    ff = fp[0]
    fp.extend([ff])
    xm, ym = Polygon(fp).buffer(-Dzkk/2).exterior.xy
    data_base.add_polyline(nice_path + "__",nice_path,0, False, True, False)
    data_base.add_coordinates(nice_path + "__", [(xx,yy) for xx,yy in zip(xm,ym)])
    
    sectors_points = get_sector_points(Dkk, Dzkk/2, N,angle_range = np.pi - (np.pi/N))
    data_base.add_polyline(nice_path + f"_sec",nice_path,0, False, True, False)
    data_base.add_coordinates(nice_path + f"_sec", [(xx,yy) for xx,yy in zip(sectors_points[0], sectors_points[1])])
   

    redraw()
    
def organizer_callback():
    dpg.configure_item("Organizer_window", show=False)
    nice_path = 'Organizer'
    iter = 1
    while 1:
        for i in data_base.get_unique_politag():
            if i == nice_path:
                nice_path = 'Organizer' + f' ({iter})'
                iter +=1
        else:
            break
    
    c = int(dpg.get_value('Column'))
    r = int(dpg.get_value('Row'))
    w = float(dpg.get_value('cell_width'))
    h = float(dpg.get_value('cell_height'))
    d = float(dpg.get_value('cell_depth'))
    t = float(dpg.get_value('thickness'))
    main_rectangle = box(0, 0,c * (w + t) + t , d)####нижняя и верхняя стенки
    main_rectangle2 = box(0, 0, d,r * (h + t) + t)
    k = int(d/20)
    wid = d/(k+0.5)
    for i in range(c+1):
        
        main_rectangle = main_rectangle.difference(box(i * (w + t), d,i * (w + t) + t , d - wid/2))
        for j in range(k):
            main_rectangle = main_rectangle.difference(box(i * (w + t), j*wid,i * (w + t) + t , j*wid + wid/2))
    Polygon_to_lines(main_rectangle,1,0,nice_path+ f'_{2}')
    dpg.add_button(label=nice_path + f'_{2}' ,parent='butonss',tag=nice_path+ f'_{2}' ,callback=active_but)
    for j in range(k):
        main_rectangle2 = main_rectangle2.difference(box(j*wid+ wid/2,0 ,(j+1)*wid  ,t ))
        main_rectangle2 = main_rectangle2.difference(box(j*wid+ wid/2,r * (h + t) + t ,(j+1)*wid  ,r * (h + t) ))
    for i in range(1,r):
        main_rectangle2 = main_rectangle2.difference(box(10,i* (h + t)   ,20,i* (h + t) +t ))
        main_rectangle2 = main_rectangle2.difference(box(d-10,i* (h + t)   ,d-20,i* (h + t) +t ))


    Polygon_to_lines(main_rectangle2,1,0,nice_path + f'_{c+1}')
    dpg.add_button(label=nice_path + f'_{c+1}' ,parent='butonss',tag=nice_path+ f'_{c+1}' ,callback=active_but)


    redraw()

def gears_callback():
    CENTER_X = float(dpg.get_value('x_center'))
    CENTER_Y = float(dpg.get_value('y_center'))
    R_del = 40.0  
    Z_zub = 34   

    gear_params = calculate_gear_parameters(R_del, Z_zub,alpha_degrees=15)
    full_gear_profile = plot_gear_profile(gear_params, num_teeth_to_show=34) 
    
    r = random.randint(1, 1000)
    dpg.add_button(label="gears"+f"{r}",parent='butonss',tag="gears"+f"{r}",callback=active_but)
    data_base.add_polyline("gears_"+f"{r}","gears"+f"{r}",0, False, True, False)
    data_base.add_coordinates("gears_"+f"{r}", full_gear_profile)
    
    redraw()
def rectangle1_callback():
    
    CENTER_X = float(dpg.get_value('center xrec'))
    CENTER_Y = float(dpg.get_value('center yrec'))
    width = float(dpg.get_value('widthrec'))
    height = float(dpg.get_value('heightrec'))
    dpg.configure_item("RECTANGLEFROMCENTER", show=False)
    r = random.randint(1, 1000)
    
    data_base.add_polyline("rec_"+f"{r}","rec"+f"{r}",0, False, True, False)
    data_base.add_coordinates("rec_"+f"{r}", [(CENTER_X-width/2,CENTER_Y-height/2),(CENTER_X-width/2,CENTER_Y+height/2),(CENTER_X+width/2,CENTER_Y+height/2),(CENTER_X+width/2,CENTER_Y-height/2),(CENTER_X-width/2,CENTER_Y-height/2)])
    
    redraw()
    dpg.add_button(label="rec"+f"{r}",parent='butonss',tag="rec"+f"{r}",callback=active_but)


def circle_callback():
    CENTER_X = float(dpg.get_value('center x'))
    CENTER_Y = float(dpg.get_value('center y'))
    radius = float(dpg.get_value('radius'))
    dpg.configure_item("CIRCLE", show=False)
    num_points = float(dpg.get_value('nps'))
    r = random.randint(1, 1000)
    dpg.add_button(label="circle"+f"{r}",parent='butonss',tag="circle"+f"{r}",callback=active_but)
    points = [
        (
            CENTER_X + radius * math.cos(2 * math.pi * i / num_points),
            CENTER_Y + radius * math.sin(2 * math.pi * i / num_points)
        )
        for i in list(range(int(num_points))) + [0]
    ]
    
    data_base.add_polyline("circle_"+f"{r}","circle"+f"{r}",0, False, True, False)
    data_base.add_coordinates("circle_"+f"{r}", points)
    
    redraw()
def test10():

    dpg.add_button(label="ruchka",parent='butonss',tag="ruchka",callback=active_but)
    data_base.add_coordinates(f"knk",[(-14,0),(-12,0),(-12,-4),(-2,-4),(-2,0),(0,0),(0,8),(4,12),(32,12),(36,8),(36,0)] + [(38,0),(38,-4),(48,-4),(48,0),(50,0),(50,14),(40,24),(-4,24),(-14,14),(-14,0)] )
    data_base.add_coordinates(f"knk2",[(-14,0),(-12,0),(-12,-4),(-2,-4),(-2,0),(0,0)] + get_circle_points(center=(4,8),radius=4,begin_angle=180,end_angle=90)+get_circle_points(center=(32,8),radius=4,begin_angle=90,end_angle=0)+ [(36,0),(38,0),(38,-4),(48,-4),(48,0),(50,0)] + get_circle_points(center=(40,14),radius=10,begin_angle=0,end_angle=90) + get_circle_points(center=(-4,14),radius=10,begin_angle=90,end_angle=180) + [(-14,0)] )
    
    # data_base.add_coordinates(f"knk",[(-14,0),(-12,0),(-12,-4),(-2,-4),(-2,0)] + get_circle_points(center=(15,0),radius=15,begin_angle=180,end_angle=90) + get_circle_points(center=(21,0),radius=15,begin_angle=90,end_angle=0) + [(38,0),(38,-4),(48,-4),(48,0),(50,0)]+ get_circle_points(center=(21,0),radius=29,begin_angle=0,end_angle=90)+ get_circle_points(center=(15,0),radius=29,begin_angle=90,end_angle=180) )
    data_base.add_polyline(f"knk","ruchka",0, False, True, False)
    data_base.add_polyline(f"knk2","ruchka",0, False, True, False)
    

  
    redraw()
def split_linestring_at_nearest_point(line_coords, target_point_coords,x,y , tolerance=1e-9):
    
    line = LineString(line_coords)
    target_point = Point(target_point_coords)
    CENTER_X = float(dpg.get_value('x_center'))
    CENTER_Y = float(dpg.get_value('y_center'))
    distance_along_line = line.project(target_point)

    split_point = line.interpolate(distance_along_line)
    
    if split_point.equals_exact(Point(line.coords[0])) or split_point.equals_exact(Point(line.coords[-1])):
        print("Ближайшая точка совпадает с началом или концом ломаной. Разделение не требуется.")
        return [line]

    # result = split(line, split_point)
    result = split(line, LineString([[x, y], [CENTER_X, CENTER_Y]]))
    if isinstance(result, GeometryCollection):
        split_lines = [geom for geom in result.geoms if isinstance(geom, LineString)]
        if len(split_lines) == 2:
            return split_lines
        else:
            
            print("split() вернул неожиданный результат. Использование ручного обрезания.")
            return cut_line(line, distance_along_line)
    
    return [line]


def cut_line(line, distance):
    """Вспомогательная функция для ручного разрезания ломаной по расстоянию."""
    if distance <= 0.0 or distance >= line.length:
        return [line]

    coords = list(line.coords)
    for i in range(len(coords) - 1):
        p1 = Point(coords[i])
        p2 = Point(coords[i+1])
        segment = LineString([p1, p2])
        segment_len = segment.length
        
        current_distance = sum(LineString(coords[j:j+2]).length for j in range(i)) + segment_len
        
        if current_distance >= distance:
           
            
            dist_on_segment = distance - (current_distance - segment_len)
            
            split_point = segment.interpolate(dist_on_segment)
            
            coords1 = coords[:i+1] + [split_point.coords[0]]
            
            coords2 = [split_point.coords[0]] + coords[i+1:]
            
            return [LineString(coords1), LineString(coords2)]

    return [line] 

def plot_mouse_click_callback():
    
    x,y = dpg.get_plot_mouse_pos()
    if dpg.get_value('change_order'):
        rec = data_base.get_all_coordinates()
        lines = []
        for i in range(len(rec)-1):
            lines.append([(rec[i][2],rec[i][3]),(rec[i+1][2],rec[i+1][3])])
        clos,i,m,_ = find_closest_pointt(lines,(x,y),set(range(len(lines))))
        tag = 1
        if m:
            tag = rec[i][1]
        else:
            tag = rec[i+1][1]
        
        if dpg.get_value('color_1'):
            data_base.update_polyline(tag,color=0,color_change_flag=1)
            
        elif dpg.get_value('color_2'):
            data_base.update_polyline(tag,color=1,color_change_flag=1)
        elif dpg.get_value('color_3'):
            data_base.update_polyline(tag,color=2,color_change_flag=1)
        elif dpg.get_value('color_4'):
            data_base.update_polyline(tag,color=3,color_change_flag=1)
        elif dpg.get_value('color_5'):
            data_base.update_polyline(tag,color=4,color_change_flag=1)


        recolor()
    elif dpg.get_value('add_text'):
        delta = 0
        val = dpg.get_value('insert_numbers')
        nice_path = 'n'+val
        iter = 1
        while 1:
            
            for i in data_base.get_unique_politag():
                if i == nice_path:
                    nice_path = 'n'+val + f' (copy {iter})'
                    iter +=1
            else: 
                break
        dpg.add_button(label=nice_path,parent='butonss',tag=nice_path,callback=active_but)
        
        heig = int(dpg.get_value('text_size'))
        kf = 35/heig
        lin_w = float(dpg.get_value('border_line_width'))
        num = 0
        for ch in val:
            
            polygon2 = chars[ch]
            counter = 0
            while 1:     
                if polygon2.geom_type == 'Polygon':
                    x1, y1 = polygon2.exterior.xy
                    coords = []
                    for h in range(len(x1)):
                        coords.append((round(x1[h]/kf+x+delta,4),  round(y1[h]/kf+y,4)))
                    for p in polygon2.interiors:
                        x1, y1 = p.xy
                        for h in range(len(x1)):
                            coords.append((round(x1[h]/kf+x+delta,4),  round(y1[h]/kf+y,4)))
                    data_base.add_polyline(nice_path+f"_{num}_"+f"{counter}",nice_path, 0, False, True, False)
                    data_base.add_coordinates(nice_path+f"_{num}_"+f"{counter}", coords)
                elif polygon2.geom_type == 'MultiPolygon':
                    counter2 = 0
                    for pol in polygon2.geoms:
                        x1, y1 = pol.exterior.xy
                        coords = []
                        for h in range(len(x1)):
                            coords.append((round(x1[h]/kf+x+delta,4),  round(y1[h]/kf+y,4)))
                        data_base.add_polyline(nice_path+f"_{num}"+f"_{counter}"+f"_{counter2}",nice_path, 0, False, True, False)
                        data_base.add_coordinates(nice_path+f"_{num}"+f"_{counter}"+f"_{counter2}", coords)
                        counter3 = 0
                        for p in pol.interiors:
                            x1, y1 = p.xy
                            coords = []
                            for h in range(len(x1)):
                                coords.append((round(x1[h]/kf+x+delta,4),  round(y1[h]/kf+y,4)))
                            data_base.add_polyline(nice_path+f"_{num}_"+f"{counter}"+f"_{counter2}"+f"_{counter3}",nice_path, 0, False, True, False)
                            data_base.add_coordinates(nice_path+f"_{num}_"+f"{counter}"+f"_{counter2}"+f"_{counter3}", coords)
                            counter3 +=1
                        counter2 +=1
                polygon2 = polygon2.buffer(-lin_w*kf,quad_segs=0)
                if polygon2.is_empty:
                    break
                counter+=1
            delta += char_shifts[ch]/kf
            num+=1
        redraw()
    elif dpg.get_value('movelines'):
        
        coords = []
        tags = data_base.get_tag_where('active=True')
        for tag in tags:
            coords += data_base.get_coordinates(tag)

        xx = [r[0] for r in coords]
        yy = [r[1] for r in coords]

        placeholders = ', '.join(f"'{t}'" for t in tags)
       
        # for tag in data_base.get_tag_where('active=True'):

        data_base.increment_field_value_with_condition(x-min(xx),y-min(yy),f'polyline_tag IN ({placeholders})')
        data_base.update_polylines(tags,redraw_flag = True)
                
        redraw()
    elif dpg.get_value('select'):
        
        rec = data_base.get_all_coordinates()
        lines = []
        for i in range(len(rec)-1):
            lines.append([(rec[i][2],rec[i][3]),(rec[i+1][2],rec[i+1][3])])
        clos,i,m,_ = find_closest_pointt(lines,(x,y),set(range(len(lines))))
        tagg = 1
        if m:
            tagg = rec[i][1]
        else:
            tagg = rec[i+1][1]
        
        active_but(data_base.get_unique_politag_where(f"tag='{tagg}'")[0])    
    
        recolor()
    elif dpg.get_value('rotate'):
        

        coords = []
        tags = data_base.get_tag_where('active=True')
        

        CENTER_X = float(dpg.get_value('x_center'))
        CENTER_Y = float(dpg.get_value('y_center'))
        
        ANGLE_RAD = np.tan(-x/y)
        

        COS_A = math.cos(ANGLE_RAD)
        SIN_A = math.sin(ANGLE_RAD)
        update_queries = []
        for tag in tags:
            points = data_base.get_coordinates_with_id(tag)

            

            for id, x, y in points:
                
                x_prime = x - CENTER_X
                y_prime = y - CENTER_Y

                x_new_prime = x_prime * COS_A - y_prime * SIN_A
                y_new_prime = x_prime * SIN_A + y_prime * COS_A

                x_new = x_new_prime + CENTER_X
                y_new = y_new_prime + CENTER_Y

                update_queries.append((x_new, y_new, id))
        data_base.update(update_queries)
        data_base.update_polylines(tags,redraw_flag = True)
        redraw()
    elif dpg.get_value('center'):
        dpg.set_value('x_center',x)
        dpg.set_value('y_center',y)
        dpg.set_value("series_center",[[x],[y]])
    elif dpg.get_value('split'):
        tags = data_base.get_tag_where('active=True')
        bigtag = data_base.get_bigtag_where('active=True')[0]
        line_coordinates = data_base.get_coordinates(tags[0])

        target_point_coordinates = (x, y) 
        polyline_segments = split_linestring_at_nearest_point(line_coordinates, target_point_coordinates,x,y)
        if polyline_segments and len(polyline_segments) == 2:

            dpg.delete_item(tags[0])
            dpg.delete_item(bigtag)
            dpg.add_button(label=bigtag+'-1',parent='butonss',tag=bigtag+'-1',callback=active_but)
            dpg.add_button(label=bigtag+'-2',parent='butonss',tag=bigtag+'-2',callback=active_but)
            placeholders = ', '.join(f"'{t}'" for t in tags)

            data_base.delete_active(f'polyline_tag IN ({placeholders})')
            
            data_base.add_polyline(bigtag+'-11',bigtag+'-1',0, False, True, False)
            data_base.add_coordinates(bigtag+'-11', polyline_segments[0].coords)
            data_base.add_polyline(bigtag+'-22',bigtag+'-2',0, False, True, False)
            data_base.add_coordinates(bigtag+'-22', polyline_segments[1].coords)
            redraw()
def call_rot():
    
    tags = data_base.get_tag_where('active=True')
    

    CENTER_X = float(dpg.get_value('x_center'))
    CENTER_Y = float(dpg.get_value('y_center'))
    rotating(tags,CENTER_X,CENTER_Y,float(dpg.get_value('rotate_angle')))
    redraw()  
def rotating(tags,centerx,centery,angle):
   

    CENTER_X = centerx
    CENTER_Y = centery
    ANGLE_RAD = math.radians(-angle)
    

    COS_A = math.cos(ANGLE_RAD)
    SIN_A = math.sin(ANGLE_RAD)
    update_queries = []
    for tag in tags:
        points = data_base.get_coordinates_with_id(tag)

        

        for id, x, y in points:
            
            x_prime = x - CENTER_X
            y_prime = y - CENTER_Y

            x_new_prime = x_prime * COS_A - y_prime * SIN_A
            y_new_prime = x_prime * SIN_A + y_prime * COS_A

            x_new = x_new_prime + CENTER_X
            y_new = y_new_prime + CENTER_Y

            update_queries.append((x_new, y_new, id))
    data_base.update(update_queries)
    data_base.update_polylines(tags,redraw_flag = True)
           
def recolor():
    for tag in data_base.get_tag('color_change_flag=True'):
        coloractive = data_base.get_color(f'tag="{tag}"')[0]
        
        if coloractive[1] == 1:
            dpg.set_value(f'color_{tag}',(255,255,255,255))
        elif coloractive[0] == 0:
            dpg.set_value(f'color_{tag}',(0, 191, 255, 255))
        elif coloractive[0] == 1:
            dpg.set_value(f'color_{tag}',(255, 20, 147, 255))
        elif coloractive[0] == 2:
            dpg.set_value(f'color_{tag}',(255, 215, 0, 255))
        elif coloractive[0] == 3:
            dpg.set_value(f'color_{tag}',(0, 255, 127, 255))
        elif coloractive[0] == 4:
            dpg.set_value(f'color_{tag}',(255, 69, 0, 255 ))
        
        data_base.update_polyline(tag,color_change_flag=False)
def scale_lines():
    tags = data_base.get_tag_where('active=True')
    
    placeholders = ', '.join(f"'{t}'" for t in tags)
    
    

    data_base.scale_value_with_condition(2,2,f'polyline_tag IN ({placeholders})')
    data_base.update_polylines(tags,redraw_flag = True)
            
    redraw() 
def redraw(all=0):


    tags = data_base.get_tag('redraw_flag=True')
    for tag in tags:

        coloractive = data_base.get_color(f'tag="{tag}"')[0]
        color = (255,255,255,255)
        if coloractive[1] == 1:
            color = (255,255,255,255)
        elif coloractive[0] == 0:
            color = (0, 191, 255, 255)
        elif coloractive[0] == 1:
            color = (255, 20, 147, 255)
        elif coloractive[0] == 2:
            color = (255, 215, 0, 255)
        elif coloractive[0] == 3:
            color = (0, 255, 127, 255)
        elif coloractive[0] == 4:
            color = (255, 69, 0, 255 )

        dpg.delete_item(f'color_{tag}')
        with dpg.theme() as coloured_line_theme1:
            with dpg.theme_component():
                dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots,tag=f'color_{tag}')
        
        coor = data_base.get_coordinates(tag)
        dpg.delete_item(f'{tag}')
        dpg.add_line_series([x for x,y in coor], [y for x,y in coor], parent=Y_AXIS_TAG,tag=tag) 
        
        dpg.bind_item_theme(dpg.last_item(), coloured_line_theme1)
    data_base.update_polylines(tags,redraw_flag = False)

def set_color():
    
    if dpg.get_value('color_1'):                 
        data_base.set_color(0)
    elif dpg.get_value('color_2'):
        data_base.set_color(1)
    elif dpg.get_value('color_3'):
        data_base.set_color(2)
    elif dpg.get_value('color_4'):
        data_base.set_color(3)
    elif dpg.get_value('color_5'):
        data_base.set_color(4) 

def delete_l():
    buts = data_base.get_unique_politag_where('active=True')
    tags = data_base.get_tag('active=True')
    for t in tags:
        dpg.delete_item(t)
    for t in buts:
        dpg.delete_item(t)
    
    placeholders = ', '.join(f"'{t}'" for t in tags)

    data_base.delete_active(f'polyline_tag IN ({placeholders})')
    redraw(1)


def split_l():
    rec = data_base.get_polyline_where('active=1')
    deleted_but = [rec[0][2]]
    dpg.delete_item(rec[0][2])
    for i,r in enumerate(rec):
        if r[2] in deleted_but:
            dpg.add_button(label=r[2]+f"_{i}",parent='butonss',tag=r[2]+f"_{i}",callback=active_but)
            data_base.update_polyline(r[1],big_tag=r[2]+f"_{i}",active=0,color_change_flag=1)
        else:
            dpg.delete_item(r[2])   
            deleted_but.append(r[2])
    recolor()
def merge_lines(lines):
    merged_result = [lines[0]]
    ss = set(range(len(lines)))
    print(ss)
    ss.remove(0)
    while ss:
        f = False
        for i in ss:
            line= lines[i]
            if not line:
                continue

            
            last_line = merged_result[-1]

            if last_line[-1] == line[0]: 
                merged_result[-1] = last_line + line  
                ss.remove(i)
                f= True
                break
            elif last_line[0] == line[-1]: 
                merged_result[-1] = line + last_line 
                ss.remove(i) 
                f= True
                break
            elif last_line[0] == line[0]:  
                merged_result[-1] = line[::-1] + last_line
                ss.remove(i)
                f= True
                break
            elif last_line[-1] == line[-1]:  
                
                merged_result[-1] = last_line + line[::-1]
                ss.remove(i)
                f= True
                break
        if not f:
            merged_result.append(line)
            ss.remove(i)


    return merged_result
def optimize_():
    data = data_base.get_cord_by_tag()
    grouped_coordinates = {}

    for tag, x, y in data:
        if tag not in grouped_coordinates:
            grouped_coordinates[tag] = []
        grouped_coordinates[tag].append((x, y))
    for t in data_base.get_all_tag():   
        dpg.delete_item(t)
    for t in data_base.get_unique_politag():
        dpg.delete_item(t)
    data_base.clear_tables()

    result = merge_lines(list(grouped_coordinates.values()))
    c = 0
    for r in result:
        data_base.add_polyline(f'r{c}','opt', 0, False, True, False)
        data_base.add_coordinates(f'r{c}', r)
        c+=1
    redraw()
    dpg.add_button(label='opt',parent='butonss',tag='opt',callback=active_but)

def rotate_x():
    invers_lines()
def bufferm():
    tags = data_base.get_tag_where('active=1')

    placeholders = ', '.join(f"'{t}'" for t in tags)
    
    
    rec = data_base.get_coordinates_where(f"polyline_tag IN ('{tags[0]}')")
    
    xx = [(r[0],r[1]) for r in rec]
    x1, y1 = Polygon(xx).buffer(-0.35).exterior.xy
    coords = []
    for h in range(len(x1)):
        coords.append((round(x1[h],4),  round(y1[h],4)))
    data_base.add_polyline(tags[0]+f"_b-",tags[0]+f"_b-", 0, False, True, False)
    data_base.add_coordinates(tags[0]+f"_b-", coords)
    dpg.add_button(label=tags[0]+f"_b-",parent='butonss',tag=tags[0]+f"_b-",callback=active_but)
    redraw()

def bufferp():
    tags = data_base.get_tag_where('active=1')

    placeholders = ', '.join(f"'{t}'" for t in tags)
    
    
    rec = data_base.get_coordinates_where(f"polyline_tag IN ('{tags[0]}')")
    
    xx = [(r[0],r[1]) for r in rec]
    x1, y1 = Polygon(xx).buffer(0.35).exterior.xy
    coords = []
    for h in range(len(x1)):
        coords.append((round(x1[h],4),  round(y1[h],4)))
    data_base.add_polyline(tags[0]+f"_b+",tags[0]+f"_b+", 0, False, True, False)
    data_base.add_coordinates(tags[0]+f"_b+", coords)
    dpg.add_button(label=tags[0]+f"_b+",parent='butonss',tag=tags[0]+f"_b+",callback=active_but)
    redraw()


def rotate_y():
    invers_lines('y')
def dublicating(bigtag):
    but = bigtag
    nice = find_nice_path(but)
    dpg.add_button(label=nice,parent='butonss',tag=nice,callback=active_but)
    lines = data_base.get_polyline_where(f'active=1 AND big_tag="{but}"')
    for i in range(len(lines)):
        coords = data_base.get_coordinates(lines[i][1])
        data_base.add_polyline(nice + f"_{i}",nice,0, False, True, False)
        data_base.add_coordinates(nice + f"_{i}", coords)
    return nice
            
        
def dublicate_lines():
    pol = data_base.get_bigtag_where('active=1')
    for but in pol:
        dublicating(but)
    redraw()
    
def move_to_center_lines():
    CENTER_X = float(dpg.get_value('x_center'))
    CENTER_Y = float(dpg.get_value('y_center'))
    coords = []
    tags = data_base.get_tag_where('active=1')
    for tag in tags:
        coords += data_base.get_coordinates(tag)

    xx = [r[0] for r in coords]
    yy = [r[1] for r in coords]

    placeholders = ', '.join(f"'{t}'" for t in tags)
    
    data_base.increment_field_value_with_condition(-min(xx)+CENTER_X,-min(yy) + CENTER_Y,f'polyline_tag IN ({placeholders})')
    data_base.update_polylines(tags,redraw_flag = True)
    redraw()
def print_coords():
    
    tags = data_base.get_tag_where('active=1')
    for tag in tags:
        print([(round(x, 2), round(y, 2)) for x, y in data_base.get_coordinates(tag)])
        
def normalize_lines():
    coords = []
    tags = data_base.get_tag_where('active=1')
    for tag in tags:
        coords += data_base.get_coordinates(tag)

    xx = [r[0] for r in coords]
    yy = [r[1] for r in coords]

    placeholders = ', '.join(f"'{t}'" for t in tags)
    
    data_base.increment_field_value_with_condition(-min(xx),-min(yy),f'polyline_tag IN ({placeholders})')
    data_base.update_polylines(tags,redraw_flag = True)
    redraw()

def invers_lines(ocb='x'):
    tags = data_base.get_tag_where('active=1')

    placeholders = ', '.join(f"'{t}'" for t in tags)
    
    
    rec = data_base.get_coordinates_where(f'polyline_tag IN ({placeholders})')
    
    xx = [r[0] for r in rec]
    yy = [r[1] for r in rec]
  
    if ocb == 'x':
        data_base.inverse_field_value_with_condition('y',max(yy) + min(yy),f'polyline_tag IN ({placeholders})')
    else:
        data_base.inverse_field_value_with_condition('x',max(xx) + min(xx),f'polyline_tag IN ({placeholders})')
    for t in tags:
        data_base.update_polyline(t,redraw_flag=1)
    redraw()


def parse_coordinates_from_string(coord_str):
    """
    Парсит координаты из строки вида "(-9.7, 1.37)" или "(-9.7, 1.37, 0)".
    Возвращает (x, y, z) или None при ошибке.
    """
    # Убираем пробелы и скобки
    coord_str = coord_str.strip().strip('()')
    if not coord_str:
        return None
    
    # Ищем числа: float с запятыми (напр. -9.7,1.37,0)
    numbers = re.findall(r'[-+]?\d*\.?\d+', coord_str)
    
    if len(numbers) == 2:  # (x, y) — 2D
        try:
            x = float(numbers[0])
            y = float(numbers[1])
            z = 0.0  # По умолчанию для 2D
            return (x, y, z)
        except ValueError:
            return None
    elif len(numbers) == 3:  # (x, y, z) — 3D
        try:
            x = float(numbers[0])
            y = float(numbers[1])
            z = float(numbers[2])
            return (x, y, z)
        except ValueError:
            return None
    else:
        print(f"Предупреждение: Некорректный формат координат '{coord_str}' (ожидалось 2 или 3 числа)")
        return None

def midpoint_between_two_points(pointA, pointB):
   
    x1, y1 = pointA[0], pointA[1]
    x2, y2 = pointB[0], pointB[1]
    
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    return (cx, cy)    

def determine_rotation_direction(O, A, B):
   
    x_A_prime = A[0] - O[0]
    y_A_prime = A[1] - O[1]
    
    x_B_prime = B[0] - O[0]
    y_B_prime = B[1] - O[1]
    
    cross_product = (x_A_prime * y_B_prime) - (x_B_prime * y_A_prime)
    return cross_product
    # if cross_product > 0:
    #     return False 
    # elif cross_product < 0:
    #     return True  
    # else:
    #     return False     
def extract_points_from_ggb(ggb_file_path):
    nice_path = find_nice_path(ggb_file_path)
    dpg.add_button(label=nice_path,parent='butonss',tag=nice_path,callback=active_but)
    points = {}
    segments = [] 
    
    try:
        with zipfile.ZipFile(ggb_file_path, 'r') as zip_ref:
            xml_content = zip_ref.read('geogebra.xml').decode('utf-8')
            
    except Exception as e:
        print(f"Ошибка при чтении ZIP: {e}")
        return points, segments
    
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Ошибка парсинга XML: {e}")
        return points, segments
    counter = 0
    for elem in root.iter('element'):
        if elem.get('type') == 'point':
            label = elem.get('label', 'unnamed')
            coords_elem = elem.find('coords')  # <coords x="..." y="..." z="..."/>
            if coords_elem is not None:
                x = float(coords_elem.get('x', 0.0))
                y = float(coords_elem.get('y', 0.0))
                points[label] = (x, y)
        for elem in root.iter('strokeCoords'):
            
            number_strings = [s.strip() for s in elem.get('val').split(',')]
        
            try:
                numbers = [float(s) for s in number_strings if s!="NaN"]
            except ValueError:
                print("Ошибка: Строка содержит нечисловые символы.")
                return []
            
            num_elements = len(numbers)
            num_pairs = num_elements // 2
            
            result_list = []
            for i in range(0, num_elements, 2):
    
                result_list.append((numbers[i], numbers[i + 1]))
                data_base.add_polyline(nice_path+f"_kar_"+f"{counter}",nice_path,0, False, True, False)
                        
                data_base.add_coordinates(nice_path+f"_kar_"+f"{counter}", result_list)
                counter+=1
    for command in root.iter('command'):
        cmd_name = command.get('name')
        



        if cmd_name == 'Segment' or cmd_name == 'Vector':
            input_elem = command.find('input')
            if input_elem is not None:
                a0 = input_elem.get('a0')  # Первая точка
                a1 = input_elem.get('a1')  # Вторая точка
                output = command.find('output')
                
                if a0 in points and a1 in points:
                    segments.append((points[a0], points[a1]))
                    data_base.add_polyline(nice_path+f"_seg_"+f"{counter}",nice_path,0, False, True, False)
                    
                    data_base.add_coordinates(nice_path+f"_seg_"+f"{counter}", [points[a0], points[a1]])
        elif cmd_name == 'Semicircle':
            input_elem = command.find('input') 
            a0 = input_elem.get('a0') 
            a1 = input_elem.get('a1')   
            c = midpoint_between_two_points(points[a0],points[a1])   
            r = distance(points[a0],points[a1])/2
            num_points = 50  
                
            start_angle = math.atan2(points[a1][1] - c[1], points[a1][0] - c[0])
    
            angles = [start_angle + (math.pi * i / (num_points + 1)) for i in range(num_points + 2)]
            
            pointss = [
                (
                    c[0] + r * math.cos(angle),
                    c[1] + r * math.sin(angle)
                )
                for angle in angles
            ]
            
            data_base.add_polyline(nice_path+f"_circle_"+f"{counter}",nice_path,0, False, True, False)
            data_base.add_coordinates(nice_path+f"_circle_"+f"{counter}", pointss) 
        elif cmd_name == 'CircleArc'or cmd_name == 'CircleSector':
            input_elem = command.find('input') 
            a0 = input_elem.get('a0') 
            a1 = input_elem.get('a1') 
            a2 = input_elem.get('a2')
            c = points[a0]
            r = distance(points[a0],points[a1])
            num_points = 20  
                
            start_angle = math.atan2(points[a1][1] - c[1], points[a1][0] - c[0])
            end_angle = math.atan2(points[a2][1] - c[1], points[a2][0] - c[0])

            angles = [start_angle + ( i * (end_angle - start_angle)/num_points) for i in range(num_points)]
            
            pointss = [
                (
                    c[0] + r * math.cos(angle),
                    c[1] + r * math.sin(angle)
                )
                for angle in angles
            ]
            if cmd_name == 'CircleSector':
                pointss.append(c)
                pointss.append(pointss[0])
            data_base.add_polyline(nice_path+f"_circle_"+f"{counter}",nice_path,0, False, True, False)
            data_base.add_coordinates(nice_path+f"_circle_"+f"{counter}", pointss) 
        elif cmd_name == 'CircumcircleArc' or cmd_name == 'CircumcircleSector':
            input_elem = command.find('input') 
            a0 = input_elem.get('a0') 
            a1 = input_elem.get('a1') 
            a2 = input_elem.get('a2')
            pointA = points[a0]
            pointB = points[a1]
            pointC = points[a2]
            xa, ya = pointA
            xb, yb = pointB
            xc, yc = pointC
            
            D = 2 * (xa * (yb - yc) + xb * (yc - ya) + xc * (ya - yb))
            
            cx = ((xa**2 + ya**2) * (yb - yc) + (xb**2 + yb**2) * (yc - ya) + (xc**2 + yc**2) * (ya - yb)) / D
            cy = ((xa**2 + ya**2) * (xc - xb) + (xb**2 + yb**2) * (xa - xc) + (xc**2 + yc**2) * (xb - xa)) / D
            
            c = (cx, cy)
            r = distance(c, pointA)  
            
            rB = distance(c, pointB)
            rC = distance(c, pointC)
            if abs(r - rB) > 1e-6 or abs(r - rC) > 1e-6:
                
                r = (r + rB + rC) / 3  

            num_points = 40  
            if  determine_rotation_direction(c,points[a0],points[a1]) > 0:

                start_angle = math.atan2(points[a0][1] - c[1], points[a0][0] - c[0])
                end_angle = 2 * np.pi + math.atan2(points[a2][1] - c[1], points[a2][0] - c[0]) 
                if start_angle < 0:
                    start_angle = 2*np.pi + start_angle
                if end_angle > 2*np.pi:
                    end_angle -= 2*np.pi
               
                print('tut')
                angles = [start_angle + ( i * (end_angle - start_angle)/num_points) for i in range(num_points+1)]
            
            else:
                start_angle = math.atan2(points[a2][1] - c[1], points[a2][0] - c[0])
                end_angle =  math.atan2(points[a0][1] - c[1], points[a0][0] - c[0])
                print('ne tut')
                angles = [start_angle + ( i * (end_angle - start_angle)/num_points) for i in range(num_points+1)]
            
            # print(start_angle,end_angle)    
            # start_angle = math.atan2(points[a0][1] - c[1], points[a0][0] - c[0])
            # end_angle = math.atan2(points[a2][1] - c[1], points[a2][0] - c[0])

            # angles = [start_angle + ( i * (end_angle - start_angle)/num_points) for i in range(num_points)]
            
            pointss = [
                (
                    c[0] + r * math.cos(angle),
                    c[1] + r * math.sin(angle)
                )
                for angle in angles
            ]
            if cmd_name == 'CircumcircleSector':
                pointss.append(c)
                pointss.append(pointss[0])
            data_base.add_polyline(nice_path+f"_circle_"+f"{counter}",nice_path,0, False, True, False)
            data_base.add_coordinates(nice_path+f"_circle_"+f"{counter}", pointss) 

        # elif cmd_name == 'CircleSector':
        #     input_elem = command.find('input') 
        #     a0 = input_elem.get('a0') 
        #     a1 = input_elem.get('a1') 
        #     a2 = input_elem.get('a2')
            
        elif cmd_name == 'Circle':

            input_elem = command.find('input')
            
            if input_elem is not None:
                center_label = input_elem.get('a0') 
                radius_input = input_elem.get('a1')  

                a2 = input_elem.get('a2')
                if a2 is not None:  # Circle[3 точки]
                    pointA = points[center_label]
                    pointB = points[radius_input]
                    pointC = points[a2]
                    xa, ya = pointA
                    xb, yb = pointB
                    xc, yc = pointC
                    
                   
                    D = 2 * (xa * (yb - yc) + xb * (yc - ya) + xc * (ya - yb))
                    
                    # Координаты центра
                    cx = ((xa**2 + ya**2) * (yb - yc) + (xb**2 + yb**2) * (yc - ya) + (xc**2 + yc**2) * (ya - yb)) / D
                    cy = ((xa**2 + ya**2) * (xc - xb) + (xb**2 + yb**2) * (xa - xc) + (xc**2 + yc**2) * (xb - xa)) / D
                    
                    center_coords = (cx, cy)
                    radius = distance(center_coords, pointA)  
                    
                    rB = distance(center_coords, pointB)
                    rC = distance(center_coords, pointC)
                    if abs(radius - rB) > 1e-6 or abs(radius - rC) > 1e-6:
                        print(f"Предупреждение: Разные радиусы (rA={radius:.6f}, rB={rB:.6f}, rC={rC:.6f})")
                        radius = (radius + rB + rC) / 3  
                    


                else:
                    center_coords = points[center_label]
                        
                    
                    radius = None
                    
                    
                    if radius_input is not None:
                        # Вариант 1: Число (напр. "2" или "-1.5")
                        if re.match(r'^[-+]?\d*\.?\d+$', radius_input):
                            
                            radius = abs(float(radius_input))
                                  
                        # Вариант 2: Сегмент (Segment[G, H])
                        elif radius_input.startswith('Segment[') and radius_input.endswith(']'):
                            seg_points_str = radius_input[8:-1] 
                            seg_points = [p.strip() for p in seg_points_str.split(',')]
                            if len(seg_points) == 2:
                                p1_label, p2_label = seg_points
                                if p1_label in points and p2_label in points:
                                    p1_coords = points[p1_label]
                                    p2_coords = points[p2_label]
                                    radius = distance(p1_coords, p2_coords)
                                    
                            
                        
                    
                        elif isinstance(radius_input, str) and radius_input.startswith('(') and radius_input.endswith(')'):
                            radius_point_coords = parse_coordinates_from_string(radius_input)
                            if radius_point_coords:
                                radius = distance(center_coords, radius_point_coords)
                                
                        
                        # Вариант 4: Label существующей точки "P"
                        elif isinstance(radius_input, str) and radius_input in points:
                            radius_point_coords = points[radius_input]
                            radius = distance(center_coords, radius_point_coords)
                            




                num_points = 50  
                
                pointss = [
                    (
                        center_coords[0] + radius * math.cos(2 * math.pi * i / num_points),
                        center_coords[1] + radius * math.sin(2 * math.pi * i / num_points)
                    )
                    for i in list(range(num_points)) + [0]
                ]
                
                data_base.add_polyline(nice_path+f"_circle_"+f"{counter}",nice_path,0, False, True, False)
                data_base.add_coordinates(nice_path+f"_circle_"+f"{counter}", pointss)   
        elif cmd_name == 'Polygon' or  cmd_name == 'PolyLine':
           
            input_elem = command.find('input')
            output = command.find('output')
            poly_label = output.get('a0') if output is not None else f"poly_{len(segments)}"
            
            if input_elem is not None:
                # Извлекаем все точки вершин (a0, a1, a2, ... — переменное число)
                vertices = []
                i = 0
                while True:
                    vertex_label = input_elem.get(f'a{i}')
                    if vertex_label is None:
                        break
                    if vertex_label in points:
                        vertices.append(vertex_label)
                    else:
                        print(f"Предупреждение: Полigon '{poly_label}' ссылается на неизвестную точку '{vertex_label}'")
                    i += 1
                
                  
                    
                data_base.add_polyline(nice_path+f"_p_"+f"{counter}",nice_path,0, False, True, False)
                for j in range(len(vertices)-1):
                    start_vertex = vertices[j]
                    end_vertex = vertices[(j + 1)] 
                    
                    segments.append((points[start_vertex], points[end_vertex]))
                    data_base.add_coordinates(nice_path+f"_p_"+f"{counter}", [points[start_vertex], points[end_vertex]])
                if cmd_name == 'Polygon':
                    start_vertex = vertices[len(vertices)-1]
                    end_vertex = vertices[0] 
                    segments.append((points[start_vertex], points[end_vertex]))
                    data_base.add_coordinates(nice_path+f"_p_"+f"{counter}", [points[start_vertex], points[end_vertex]])
        else:
            print(cmd_name + "asd")    
        counter+=1            
    
    return points, segments

def extract_coordinates2(d_string):
    # Регулярное выражение для поиска команд и координат
    pattern = re.compile(r'([MLHVCSZ])([^MLHVCSZ]*)')
    coordinates = []
    current_position = (0, 0)  # Начальная позиция (можно установить на произвольные значения)
    
    for command, params in pattern.findall(d_string):
        # Преобразование параметров в координаты
        params = params.strip().split()
        if command in 'ML':
            # Absolute move (M) или line (L)
            for i in range(0, len(params), 2):
                x = float(params[i])
                y = float(params[i + 1])
                coordinates.append((x, y))
                current_position = (x, y)
        elif command in 'HV':  # Horizontal (H) or Vertical (V)
            for param in params:
                if command == 'H':
                    x = float(param)
                    coordinates.append((x, current_position[1]))
                    current_position = (x, current_position[1])
                elif command == 'V':
                    y = float(param)
                    coordinates.append((current_position[0], y))
                    current_position = (current_position[0], y)
        elif command == 'C':  # Cubic Bézier Curve
            # Параметры для кривой: C x1, y1, x2, y2, x, y
            for i in range(0, len(params), 6):
                x1 = float(params[i])
                y1 = float(params[i + 1])
                x2 = float(params[i + 2])
                y2 = float(params[i + 3])
                x = float(params[i + 4])
                y = float(params[i + 5])
                coordinates.append((x1, y1))
                coordinates.append((x2, y2))
                coordinates.append((x, y))
                current_position = (x, y)
    
    return coordinates

def extract_points_from_svg(path):
    nice_path = find_nice_path(path)
    dpg.add_button(label=nice_path,parent='butonss',tag=nice_path,callback=active_but)
    tree = ET.parse(path)
    root = tree.getroot()
    counter = 0
    
    for element in root.iter():
        if element.tag.endswith('path') and 'd' in element.attrib:
            d_string = element.attrib['d']
            
            # xx,yy,ccx,ccy = extract_coordinates(d_string,0,0)
            ass = extract_coordinates2(d_string)
            data_base.add_polyline(nice_path+f"{counter}",nice_path,0, False, True, False)
                    # [(x,y) for x,y in zip(xx,yy)]
            data_base.add_coordinates(nice_path+f"{counter}", ass)
        elif 'x' in element.attrib and 'y' in element.attrib:
            x = float(element.attrib['x'])
            y = float(element.attrib['y'])
            print(f"Element - ID: {element.attrib.get('id', 'N/A')}, x: {x}, y: {y}")
        counter +=1

def extract_points_from_gcode(path):
    gcode = ''
    with open(path, 'r') as file:
        gcode = file.read()
    nice_path = find_nice_path(path)
    dpg.add_button(label=nice_path,parent='butonss',tag=nice_path,callback=active_but)
    lines = gcode.strip().split('\n')
    paths = []
    current_path = []

    for line in lines:
        if line.startswith('G0') or line.startswith('G1') or line.startswith('X') :
           
            parts = line.split()
            coords = {}
            for part in parts:
                if part.startswith('X'):
                    coords['X'] = float(part[1:])
                    
                elif part.startswith('Y'):
                    coords['Y'] = float(part[1:])
                    
            if 'X' in coords and 'Y' in coords:
                current_path.append((coords['X'], coords['Y']))
        elif line.startswith('S0'):
          
            if current_path:
                
                paths.append(current_path)
                current_path = []

    if current_path:
        paths.append(current_path)
    counter = 0
    for p in paths:
        # print(p)
        data_base.add_polyline(nice_path+f"{counter}",nice_path,0, False, True, False)
                    
        data_base.add_coordinates(nice_path+f"{counter}", p)
        counter +=1

def find_nice_path(path):
    nice_path = path
    iterr = 1
    while 1:
        for i in data_base.get_unique_politag():
            if i == nice_path:
                nice_path = path + f' (copy {iterr})'
                iterr +=1
        else: 
            break
    return nice_path

def pr(selected_files):
    global esyedaflag
    current_file = selected_files[0]
    if dpg.get_value('eraseold'):
        for t in data_base.get_all_tag():
            
            dpg.delete_item(t)
        for t in data_base.get_unique_politag():
            dpg.delete_item(t)
        data_base.clear_tables()
        dpg.delete_item(Y_AXIS_TAG, children_only=True, slot=1)
        dpg.add_scatter_series([float(dpg.get_value('x_center'))], [float(dpg.get_value('y_center'))], parent=Y_AXIS_TAG, tag="series_center")
        dpg.bind_item_theme("series_center", "plot_theme")
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
            
        else:
            # read_dxf_lines(current_file)
            read_dxf_with_grabber(current_file)
            redraw()
    elif '.png' in current_file:   
        lines = extract_black_lines(current_file,0.1)
        nice_path = os.path.basename(current_file)
        iterr = 1
        while 1:
            for i in data_base.get_unique_politag():
                if i == nice_path:
                    nice_path = os.path.basename(current_file) + f' (copy {iterr})'
                    iterr +=1
            else: 
                break
        counter = 0
        for l in lines:
            data_base.add_polyline(nice_path+f"{counter}",nice_path,0, False, True, False)
            data_base.add_coordinates(nice_path+f"{counter}", [(l[0],l[1]),(l[2],l[3])])
            counter +=1
        redraw()
    elif ".ggb" in current_file:
        points, segm = extract_points_from_ggb(current_file)
        redraw()
    elif ".svg" in current_file:
        extract_points_from_svg(current_file)
        redraw()
    elif ".gcode" in current_file:
        extract_points_from_gcode(current_file)
        redraw()
def generate_gear_profile(Rd, Z, alpha_deg=20, num_points=20):
   
    alpha_rad = math.radians(alpha_deg) 
    
    m = 2 * Rd / Z
    
    Rb = Rd * math.cos(alpha_rad)      # Радиус основной окружности (Base)
    Ra = Rd + m                        # Радиус окружности вершин (Addendum)
    Rf = Rd - 1.25 * m                 # Радиус окружности впадин (Dedendum)

    pitch_angle = 2 * math.pi / Z 
    
    tooth_thickness_angle = math.pi / Z 
    
    def evolute_angle_from_radius(R):
        
        if R < Rb:
            return 0.0
        
        return math.sqrt((R / Rb)**2 - 1)
    
    def involute_coordinates(R_b, theta):
     
        x = R_b * (math.cos(theta) + theta * math.sin(theta))
        y = R_b * (math.sin(theta) - theta * math.cos(theta))
        return x, y

    theta_start = 0.0 
    theta_end = evolute_angle_from_radius(Ra)
    
    profile_points = []
    
    for i in range(num_points + 1):
        
        theta = theta_start + (theta_end - theta_start) * (i / num_points)
        
        x_base, y_base = involute_coordinates(Rb, theta)
        
        involute_start_angle = (tooth_thickness_angle / 2) - (math.tan(alpha_rad) - alpha_rad)
        
        angle = involute_start_angle - theta 
        
        x = x_base * math.cos(angle) - y_base * math.sin(angle)
        y = x_base * math.sin(angle) + y_base * math.cos(angle)
        
        if math.sqrt(x**2 + y**2) >= Rf:
            profile_points.append((x, y))


    full_profile = []
    
  
    start_angle = -pitch_angle / 2 + tooth_thickness_angle / 2
    
    x_start = Rf * math.cos(start_angle)
    y_start = Rf * math.sin(start_angle)
    
    full_profile.append((x_start, y_start))
    
    full_profile.extend(profile_points)

    x_peak_right, y_peak_right = full_profile[-1]
    
    peak_angle = 2 * involute_start_angle
    
    peak_step_angle = peak_angle / (num_points // 2) 
    
    current_angle = math.atan2(y_peak_right, x_peak_right)
    
    for i in range(1, num_points // 2 + 1):
        current_angle -= peak_step_angle
        x = Ra * math.cos(current_angle)
        y = Ra * math.sin(current_angle)
        full_profile.append((x, y))

    left_profile = []
    for x, y in profile_points:
        left_profile.append((-x, y))
        
    full_profile.extend(left_profile[::-1])
    
    x_end_left, y_end_left = full_profile[-1]
    
    trough_center_angle = -pitch_angle / 2
    
    trough_step_angle = (math.atan2(y_start, x_start) - math.atan2(y_end_left, x_end_left)) / (num_points // 2)
    
    current_angle = math.atan2(y_end_left, x_end_left)
    
    for i in range(1, num_points // 2 + 1):
        current_angle -= trough_step_angle
        x = Rf * math.cos(current_angle)
        y = Rf * math.sin(current_angle)
        full_profile.append((x, y))

    if full_profile[-1] != full_profile[0]:
        full_profile.append(full_profile[0])
        
    return full_profile

def joinsel_callback():
    
    lines = data_base._extract_lines()
    
    if not lines:
        print("Нет ломаных линий для слияния.")
        return []
    # dolb = 0
    # while True:
    #     print(dolb)
    #     dolb +=1
    #     best_match = None
    #     min_distance = 0.1
        
    #     for i in range(len(lines)):
    #         for j in range(i + 1, len(lines)):
    #             line1 = lines[i]
    #             line2 = lines[j]
                
    #             end1_start = line1[0]
    #             end1_end = line1[-1]
                
    #             end2_start = line2[0]
    #             end2_end = line2[-1]
                
    #             dist_1e_2s = distance(end1_end, end2_start)
                
    #             if dist_1e_2s < min_distance:
    #                 min_distance = dist_1e_2s

    #                 best_match = (i, j, 'end-start', dist_1e_2s)
                    
                    
    #             dist_1e_2e = distance(end1_end, end2_end)
    #             if dist_1e_2e < min_distance:
    #                 min_distance = dist_1e_2e
    #                 best_match = (i, j, 'end-end', dist_1e_2e)
                    
    #             dist_1s_2s = distance(end1_start, end2_start)
    #             if dist_1s_2s < min_distance:
    #                 min_distance = dist_1s_2s
    #                 best_match = (i, j, 'start-start', dist_1s_2s)
                    
    #             dist_1s_2e = distance(end1_start, end2_end)
    #             if dist_1s_2e < min_distance:
    #                 min_distance = dist_1s_2e
                    
    #                 best_match = (j, i, 'end-start', dist_1s_2e) 
                    
    #     if not best_match:
    #         break

        
    #     i, j, orientation, distance_ = best_match
        
    #     line_a = lines[i]
    #     line_b = lines[j]
    #     new_line = []
    #     if orientation == 'end-start':
    #         new_line = line_a + line_b
    #     elif orientation == 'end-end':
    #         new_line = line_a + line_b[::-1]
    #     elif orientation == 'start-start':
    #         new_line = line_a[::-1] + line_b
        
    #     lines_to_remove = sorted([i, j], reverse=True)
    #     for index in lines_to_remove:
    #         lines.pop(index)
            
    #     lines.append(new_line)
        
    # buts = data_base.get_unique_politag_where('active=True')
    # tags = data_base.get_tag('active=True')
    # for t in tags:
    #     dpg.delete_item(t)
    # for t in buts:
    #     dpg.delete_item(t)
    
    # placeholders = ', '.join(f"'{t}'" for t in tags)

    # data_base.delete_active(f'polyline_tag IN ({placeholders})')
    # c = 1
    
    # for line in lines:
    #     if distance(line[0],line[-1]) < 0.5:
    #         line.append(line[0])
    #     data_base.add_polyline(f'joinn{c}',f'join{c}',0, False, True, False)
    #     data_base.add_coordinates(f'joinn{c}', line)
    #     dpg.add_button(label=f'join{c}',parent='butonss',tag=f'join{c}',callback=active_but)
    #     c+=1
    # redraw()
   
def join_callback():
    
    results = data_base.merge_polylines()
    tags = data_base.get_polylines_tag()
    
    for t in tags:
        
        dpg.delete_item(f'{t[0]}')
    but = data_base.get_unique_politag()
    for b in but:
        
        dpg.delete_item(f'{b[0]}')
    data_base.clear_tables()
    dpg.add_button(label='join',parent='butonss',tag='join',callback=active_but)
    for i, line in enumerate(results):
        
        data_base.add_polyline(f"{i+1}{line.geom_type}",'join',0, False, True, False)
        data_base.add_coordinates(f"{i+1}{line.geom_type}", list(line.coords))
        
        
    redraw()


        
def check_com_callback():
    ports = serial.tools.list_ports.comports()
    dpg.delete_item('com_tag')
    dpg.add_combo(label="Port", items=[port.device for port in ports],width=60,tag='com_tag',parent='forcombo')
    
def load_gcode_callback():

    gcode_commands = dpg.get_value('multiline_input').split('\n')
    
    try:
       
        for line in gcode_commands:
            if line != '':
                if laser.send_command(line):
                    dpg.set_value("multiline_input2",dpg.get_value("multiline_input2")+f"\nyes {line}")
                else:
                    dpg.set_value("multiline_input2",dpg.get_value("multiline_input2")+f"\nno  {line}")

    except serial.SerialException as e:
        print(f"Ошибка последовательного порта: {e}")

    

def calback_but1():
    gcode_l = 'G28\n'
    dpg.set_value('multiline_input', gcode_l)
def calback_but2():
    gcode_l = 'G91\nG1 X0 Y10 F1000\nG90\n'
    dpg.set_value('multiline_input', gcode_l)
def calback_but3():
    return
def calback_but4():
    gcode_l = 'G91\nG1 X-10 Y0 F1000\nG90\n'
    dpg.set_value('multiline_input', gcode_l)
def calback_but5():
    gcode_l = 'G92 X0 Y0\n'
    dpg.set_value('multiline_input', gcode_l)
def calback_but6():
    gcode_l = 'G91\nG1 X10 Y0 F1000\nG90\n'
    dpg.set_value('multiline_input', gcode_l)
def calback_but7():
    return
def calback_but8():
    gcode_l = 'G91\nG1 X0 Y-10 F1000\nG90\n'
    dpg.set_value('multiline_input', gcode_l)

def calculate_point(reference_point, angle_degrees, distance):
    angle_radians = math.radians(angle_degrees)
    new_x = reference_point[0] + distance * math.cos(angle_radians)
    new_y = reference_point[1] + distance * math.sin(angle_radians)
    
    return (new_x, new_y) 
def calback_but9():
    
    return
def izgib_callback():
    dpg.configure_item("IZGIB", show=False)
    a = float(dpg.get_value('a'))
    b = float(dpg.get_value('b'))
    c = float(dpg.get_value('c'))
    d = float(dpg.get_value('d'))
    numlins = int(dpg.get_value('numlins'))
    w = float(dpg.get_value('w'))


    dpg.add_button(label="col",parent='butonss',tag="col",callback=active_but)
    data_base.add_coordinates(f"5", [(0,0),(a+b+c,0),(a+b+c,d),(0,d),(0,0)])
    data_base.add_polyline(f"5","col",0, False, True, False)
    lenlin1 = (d-(numlins-1) * w)/(numlins-1)
    lenlin2 = (d-(numlins) * w)/(numlins-1)
    mas1 = [0,lenlin1/2,lenlin1/2+w]
    mas2 = [w,w+lenlin2]
    
    if numlins > 2:
        for i in range(numlins-2):
            mas1.append(mas1[-1] + lenlin1)
            mas1.append(mas1[-1] + w)
            mas2.append(mas2[-1] + w)
            mas2.append(mas2[-1] + lenlin2)
    mas1.append(d)
    for col in range(round(b/w)):
        if col != 0:
            for i in range(numlins):
                data_base.add_polyline(f"{col}_{i}","col",0, False, True, False)
                data_base.add_coordinates(f"{col}_{i}", [(col*w+a,mas1[i*2]),(col*w+a,mas1[i*2+1])])
        for i in reversed(range(numlins-1)):
            data_base.add_polyline(f"{col}_{i+numlins}","col",0, False, True, False)
            data_base.add_coordinates(f"{col}_{i+numlins}", [(col*w+w/2+a,mas2[i*2]),(col*w+w/2+a,mas2[i*2+1])])
    
    redraw()




def load_all():
    
    tag0 = data_base.get_tag_where('color=0')
    tag1 = data_base.get_tag_where('color=1')
    tag2 = data_base.get_tag_where('color=2')
    tag3 = data_base.get_tag_where('color=3')
    tag4 = data_base.get_tag_where('color=4')

    gcode_lines = get_gcode_from_tags([tag0,tag1,tag2,tag3,tag4])
    
    dpg.set_value('multiline_input',"\n".join(gcode_lines))

def load_sel():
    
    tag0 = data_base.get_tag_where('color=0 AND active=True')
    tag1 = data_base.get_tag_where('color=1 AND active=True')
    tag2 = data_base.get_tag_where('color=2 AND active=True')
    tag3 = data_base.get_tag_where('color=3 AND active=True')
    tag4 = data_base.get_tag_where('color=4 AND active=True')

    gcode_lines = get_gcode_from_tags([tag0,tag1,tag2,tag3,tag4])
    
    dpg.set_value('multiline_input',"\n".join(gcode_lines))




def konus_callback():
    dpg.configure_item("KONUS", show=False)
    l = float(dpg.get_value('l'))
    L = float(dpg.get_value('L'))
    d = float(dpg.get_value('d_'))
    num_points = int(dpg.get_value('numpoints'))
    a = float(dpg.get_value('a_'))
    w = float(dpg.get_value('w_'))
    numlins = int(dpg.get_value('numlins_'))
    radius = (d*l)/(L-l)
    radius2 = radius + d
    

    angle_degrees = (l*180)/(np.pi * radius)
    
    angles = np.linspace(np.pi, np.deg2rad(angle_degrees)+np.pi, num_points)

    points = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in reversed(angles)]
    points2 = [(radius2 * np.cos(angle), radius2 * np.sin(angle)) for angle in angles]
    lenlin1 = (d-(numlins-1) * w)/(numlins-1)
    lenlin2 = (d-(numlins) * w)/(numlins-1)
    mas1 = [0,lenlin1/2,lenlin1/2+w]
    mas2 = [w,w+lenlin2]
    
    if numlins > 2:
        for i in range(numlins-2):
            mas1.append(mas1[-1] + lenlin1)
            mas1.append(mas1[-1] + w)
            mas2.append(mas2[-1] + w)
            mas2.append(mas2[-1] + lenlin2)
    mas1.append(d)
    T = []
    TT = []
    for m in mas1:
        T.append([((radius + m) * np.cos(angle), (radius + m) * np.sin(angle)) for angle in angles])
    for m in mas2:
        TT.append([((radius + m) * np.cos(angle), (radius + m) * np.sin(angle)) for angle in angles])
    dpg.add_button(label="arc",parent='butonss',tag="arc",callback=active_but)
    # data_base.add_coordinates(f"1", points + [(points[len(points)-1][0],points[len(points)-1][1]+a),((points2[0][0],points2[0][1]+a))] + points2 + [calculate_point(points2[len(points)-1],-90 + angle_degrees,a),calculate_point(points[0],-90 + angle_degrees,a),points[0]])
    print(angle_degrees)
    pp = points
    pp.append(calculate_point(pp[-1],90 ,12))
    pp.append(calculate_point(pp[-1],180,4))
    pp.append(calculate_point(pp[-1],90 ,18))
    pp.append(calculate_point(pp[-1],0,4))
    pp.append(calculate_point(pp[-1],90 ,10))
    pp.append(calculate_point(pp[-1],180,d))
    pp.append(calculate_point(pp[-1],-90 ,10))
    pp.append(calculate_point(pp[-1],0,4))
    pp.append(calculate_point(pp[-1],-90 ,18))
    pp.append(calculate_point(pp[-1],180,4))

    pp = pp + points2
    pp.append(calculate_point(pp[-1],-90 + angle_degrees,5))
    pp.append(calculate_point(pp[-1],angle_degrees,4))
    pp.append(calculate_point(pp[-1],-90 + angle_degrees,15))
    pp.append(calculate_point(pp[-1],angle_degrees-180,4))
    pp.append(calculate_point(pp[-1],-90 + angle_degrees,20))
    pp.append(calculate_point(pp[-1], angle_degrees,d))
    pp.append(calculate_point(pp[-1],+90 + angle_degrees,20))
    pp.append(calculate_point(pp[-1],angle_degrees-180,4))
    pp.append(calculate_point(pp[-1],+90 + angle_degrees,15))
    pp.append(calculate_point(pp[-1],angle_degrees,4))
    pp.append(calculate_point(pp[-1],+90 + angle_degrees,5))
    data_base.add_coordinates(f"11111", pp)
   
    data_base.add_polyline(f"11111","arc",0, False, True, False)
    
    for i in range(len(T[0])):
        if i %2 == 0:
            for j in range(numlins):
                data_base.add_coordinates(f"{T[j*2][i]}_{T[j*2+1][i]}", [T[j*2+1][i],T[j*2][i]])
                data_base.add_polyline(f"{T[j*2][i]}_{T[j*2+1][i]}","arc",0, False, True, False)
            
            
        else:
            for j in reversed(range(numlins-1)):
                data_base.add_coordinates(f"{TT[j*2][i]}_{TT[j*2+1][i]}", [TT[j*2][i],TT[j*2+1][i]])
                data_base.add_polyline(f"{TT[j*2][i]}_{TT[j*2+1][i]}","arc",0, False, True, False)


            

    

    redraw()
def horizont_callback():
    dpg.add_button(label="horizont",parent='butonss',tag="horizont",callback=active_but)
    w = float(dpg.get_value('border_line_width'))
    
    points1 = [(0,i*w)for i in range(round(10/w))]
    points2 = [(10,i*w)for i in range(round(10/w))]
    for i in range(0,len(points1),2):
        
        data_base.add_coordinates(f"{i}horizont", [points1[i],points2[i]])
        data_base.add_coordinates(f"{i+1}horizont", [points2[i+1],points1[i+1]])
        data_base.add_polyline(f"{i}horizont","horizont",0, False, True, False)
        data_base.add_polyline(f"{i+1}horizont","horizont",0, False, True, False)

    redraw()

def vertical_callback(): 
    dpg.add_button(label="vertical",parent='butonss',tag="vertical",callback=active_but)
    w = float(dpg.get_value('border_line_width'))
    
    points1 = [(i*w,0)for i in range(round(10/w))]
    points2 = [(i*w,10)for i in range(round(10/w))]
    for i in range(0,len(points1),2):
        
        data_base.add_coordinates(f"{i}vertical", [points1[i],points2[i]])
        data_base.add_coordinates(f"{i+1}vertical", [points2[i+1],points1[i+1]])
        data_base.add_polyline(f"{i}vertical","vertical",0, False, True, False)
        data_base.add_polyline(f"{i+1}vertical","vertical",0, False, True, False)

    redraw()
def diagonal_callback(): 
    dpg.add_button(label="diagonal",parent='butonss',tag="diagonal",callback=active_but)
    w = float(dpg.get_value('border_line_width')) * 1.4142
    
    points1 = [(i*w,0)for i in range(1,round(10/w))]
    points2 = [(0,i*w)for i in range(1,round(10/w))]

    points11 = [(i*w,10)for i in reversed(range(round(10/w)))]
    points22 = [(10,i*w)for i in reversed(range(round(10/w)))]

    for i in range(0,len(points1),2):
        
        data_base.add_coordinates(f"{i}diagonal", [points1[i],points2[i]])
        data_base.add_coordinates(f"{i+1}diagonal", [points2[i+1],points1[i+1]])
        data_base.add_polyline(f"{i}diagonal","diagonal",0, False, True, False)
        data_base.add_polyline(f"{i+1}diagonal","diagonal",0, False, True, False)

    for i in range(0,len(points1),2):
        
        data_base.add_coordinates(f"{i}_diagonal", [points11[i],points22[i]])
        data_base.add_coordinates(f"{i+1}_diagonal", [points22[i+1],points11[i+1]])
        data_base.add_polyline(f"{i}_diagonal","diagonal",0, False, True, False)
        data_base.add_polyline(f"{i+1}_diagonal","diagonal",0, False, True, False)
        
    redraw()
def add_circle_to_plot(center=(0,0),radius=1,begin_angle=0,end_angle=360,name='name',parent='parent'):
    angles = np.linspace(np.deg2rad(begin_angle), np.deg2rad(end_angle), 40)
    points = [(radius * np.cos(angle) + center[0], radius * np.sin(angle)+ center[1]) for angle in angles]
    data_base.add_coordinates(name, points)
    data_base.add_polyline(name,parent,0, False, True, False)
def get_circle_points(center=(0,0),radius=1,begin_angle=0,end_angle=360):
    angles = np.linspace(np.deg2rad(begin_angle), np.deg2rad(end_angle), 40)
    points = [(radius * np.cos(angle) + center[0], radius * np.sin(angle)+ center[1]) for angle in angles]
    return points
def rotate_point(point, angle, center):
    angle_rad = math.radians(angle)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    # Смещение точки относительно центра
    translated_x = point[0] - center[0]
    translated_y = point[1] - center[1]

    # Применение вращения
    rotated_x = translated_x * cos_angle - translated_y * sin_angle + center[0]
    rotated_y = translated_x * sin_angle + translated_y * cos_angle + center[1]

    return (rotated_x, rotated_y)

def get_rectangle_points_from_center(width,height,center=(0,0),angle=0):
    points = [(center[0]-width/2,center[1]-height/2),
            (center[0]-width/2,center[1]+height/2),
            (center[0]+width/2,center[1]+height/2),
            (center[0]+width/2,center[1]-height/2),
            (center[0]-width/2,center[1]-height/2)]
    if angle != 0:
        points = [rotate_point(point, angle, center) for point in points]
        
    return points
def get_rectangle_points_from_left_bot(width,height,lb=(0,0),angle=0):
    points = [(lb[0],lb[1]),
            (lb[0]+width,lb[1]),
            (lb[0]+width,lb[1]+height),
            (lb[0],lb[1]+height),
            (lb[0],lb[1])]
    if angle != 0:
        points = [rotate_point(point, angle, lb) for point in points]
        
    return points
def place_in_a_circle_callback():
    centerx = float(dpg.get_value('centerxplace'))
    centery = float(dpg.get_value('centeryplace'))
    count = float(dpg.get_value('countforspace'))
    spaser = dpg.get_value('varrast')
    print(spaser)
    degrr = float(dpg.get_value('rastdegrees'))
    dpg.configure_item("place_in_a_circle", show=False)
    obj = dpg.get_value('activetext').split("\n")
    ang = degrr if spaser =='spacing' else degrr/count
    
    for o in obj:
        if o!= "":
            for i in range(int(count)-1):
                no = dublicating(o)

                rotating(data_base.get_tag(f"big_tag='{no}'"),centerx=centerx,centery=centery,angle=ang * (i+1))
            

    redraw()
    
def tangent_at_point(point1, point2):
        return (point2[0] - point1[0], point2[1] - point1[1])


# def get_pol_for_laser(polygon):
#     # polygon = Polygon([(x,y)for x,y in zip(xx,yy)])

#     points = np.array(polygon.exterior.coords)

#     tangents = []
#     perpendiculars = []
#     length = 0.5

#     for i in range(len(points) - 1):
#         p1 = points[i]
#         p2 = points[i + 1]
#         tangent = tangent_at_point(p1, p2)
#         tangents.append((p1, tangent))
        
#         # Вычисляем перпендикулярный вектор
#         perp_vector = (-tangent[1], tangent[0])
#         angle_radians = np.arctan2(perp_vector[1], perp_vector[0])

#         angle_degrees = np.degrees(angle_radians)
#         get_radius(0.2,0.2,perp_vector[0],perp_vector[1])
#         # Нормализация перпендикуляра
#         norm = np.sqrt(perp_vector[0]**2 + perp_vector[1]**2)
#         perp_vector = (perp_vector[0] / norm * length, perp_vector[1] / norm * length)
        
#         # Конечная точка перпендикуляра
#         end_point = (p1[0] + perp_vector[0], p1[1] + perp_vector[1])
#         perpendiculars.append((p1, end_point))


def test2():

    dpg.add_button(label="kn",parent='butonss',tag="kn",callback=active_but)
    
    points =  get_circle_points(center=(32,32),radius=31,begin_angle=0,end_angle=360)
    
    data_base.add_coordinates(f"knk", points)
    data_base.add_polyline(f"knk","kn",0, False, True, False)
    points2 =  get_circle_points(center=(32,32),radius=4.5,begin_angle=0,end_angle=360)
    
    data_base.add_coordinates(f"knk2", points2)
    data_base.add_polyline(f"knk2","kn",0, False, True, False)


    points12 =  get_circle_points(center=(32+16.5,32),radius=0.4,begin_angle=0,end_angle=360)
    points13 =  get_circle_points(center=(32,32-16.5),radius=0.4,begin_angle=0,end_angle=360)
    points14 =  get_circle_points(center=(32-16.5,32),radius=0.4,begin_angle=0,end_angle=360)
    points15 =  get_circle_points(center=(32,32+16.5),radius=0.4,begin_angle=0,end_angle=360)
    points16 =  get_circle_points(center=(32+13,32+13),radius=5,begin_angle=0,end_angle=360)
    data_base.add_coordinates(f"knk12", points12)
    data_base.add_polyline(f"knk12","kn",0, False, True, False)
    data_base.add_coordinates(f"knk13", points13)
    data_base.add_polyline(f"knk13","kn",0, False, True, False)
    data_base.add_coordinates(f"knk14", points14)
    data_base.add_polyline(f"knk14","kn",0, False, True, False)
    data_base.add_coordinates(f"knk15", points15)
    data_base.add_polyline(f"knk15","kn",0, False, True, False)
    data_base.add_coordinates(f"knk16", points16)
    data_base.add_polyline(f"knk16","kn",0, False, True, False)

    redraw()
def calculate_new_point(x1, y1, distance, angle_degrees):
    
    angle_radians = math.radians(angle_degrees)
    
    x2 = x1 + distance * math.cos(angle_radians)
    y2 = y1 + distance * math.sin(angle_radians)
    
    return x2, y2
def calculate_new_point2(xy, distance, angle_degrees):
    x1,y1 = xy
    angle_radians = math.radians(angle_degrees)
    
    x2 = x1 + distance * math.cos(angle_radians)
    y2 = y1 + distance * math.sin(angle_radians)
    
    return x2, y2
def find_intersection(p1, p2, p3, p4):
    """
    Находит точку пересечения двух линий, заданных двумя точками каждая.
    
    :param p1: Координаты первой точки первой линии (x1, y1)
    :param p2: Координаты второй точки первой линии (x2, y2)
    :param p3: Координаты первой точки второй линии (x3, y3)
    :param p4: Координаты второй точки второй линии (x4, y4)
    :return: Координаты точки пересечения (x, y) или None, если линии параллельны
    """
    
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Вычисление переменных для уравнения
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # Если denominator равен 0, линии параллельны
    if denominator == 0:
        return None
    
    # Находится точка пересечения
    intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    
    return intersection_x, intersection_y
def test4():
    dpg.add_button(label="bottom",parent='butonss',tag="bottom",callback=active_but)
    dpg.add_button(label="top",parent='butonss',tag="top",callback=active_but)
    big_c =  get_circle_points(center=(0,0),radius=35,begin_angle=0,end_angle=360)
    r1 =  get_circle_points(center=(-80,0),radius=20,begin_angle=0,end_angle=360)
    r4 =  get_circle_points(center=(80,0),radius=20,begin_angle=0,end_angle=360)
    data_base.add_coordinates(f"bc", big_c)
    data_base.add_polyline(f"bc","bottom",0, False, True, False)
    data_base.add_coordinates(f"r1", r1)
    data_base.add_polyline(f"r1","bottom",0, False, True, False)
    data_base.add_coordinates(f"r4", r4)
    data_base.add_polyline(f"r4","bottom",0, False, True, False)

    c2x,c2y = calculate_new_point(0,0,80,-120)
    r2 =  get_circle_points(center=(c2x,c2y),radius=20,begin_angle=0,end_angle=360)
    data_base.add_coordinates(f"r2", r2)
    data_base.add_polyline(f"r2","bottom",0, False, True, False)

    c3x,c3y = calculate_new_point(0,0,80,-60)
    r3 =  get_circle_points(center=(c3x,c3y),radius=20,begin_angle=0,end_angle=360)
    data_base.add_coordinates(f"r3", r3)
    data_base.add_polyline(f"r3","bottom",0, False, True, False)

    x,y = calculate_new_point(80,0,35,-30)
    x1,y1 = calculate_new_point(x,y,12,240)
    x4,y4 = calculate_new_point(x,y,30,240)
    x2,y2 = calculate_new_point(x1,y1,4,-30)
    x3,y3 = calculate_new_point(x4,y4,4,-30)

    x,y = calculate_new_point(c3x,c3y,35,-30)
    x11,y11 = calculate_new_point(x,y,30,60)
    x41,y41 = calculate_new_point(x,y,12,60)
    x21,y21 = calculate_new_point(x11,y11,4,-30)
    x31,y31 = calculate_new_point(x41,y41,4,-30)
    x,y = calculate_new_point(c3x,c3y,35,-90)
    x12,y12 = (x-5,y)
    x42,y42 = (x-20,y)
    x22,y22 = (x12,y12-4)
    x32,y32 = (x42,y42-4)
    contur = [(-80,35),(-80+15,35),(-80+15,35+4),(-80+45,35+4),(-80+45,35),(80-45,35),(80-45,35+4),(80-15,35+4),(80-15,35)] + get_circle_points(center=(80,0),radius=35,begin_angle=90,end_angle=-30) + [(x1,y1),(x2,y2),(x3,y3),(x4,y4),(x11,y11),(x21,y21),(x31,y31),(x41,y41)] + get_circle_points(center=(c3x,c3y),radius=35,begin_angle=-30,end_angle=-90) + [(x12,y12),(x22,y22),(x32,y32),(x42,y42)] + [(-x12,y12),(-x22,y22),(-x32,y32),(-x42,y42)][::-1] + get_circle_points(center=(c2x,c2y),radius=35,begin_angle=270,end_angle=210)+ [(-x1,y1),(-x2,y2),(-x3,y3),(-x4,y4),(-x11,y11),(-x21,y21),(-x31,y31),(-x41,y41)][::-1] + get_circle_points(center=(-80,0),radius=35,begin_angle=210,end_angle=90) 
    data_base.add_coordinates(f"contur", contur)
    data_base.add_polyline(f"contur","bottom",0, False, True, False)
    


    
    radius = 48.35

    x,y = find_intersection((-80,35),(80,35),(calculate_new_point(-80,0,55,210)),(calculate_new_point(c2x,c2y,55,210)))
    bot_cx, bot_cy = calculate_new_point(x,y,radius*2,-30)
   
    x1b,y1b = calculate_new_point(x1,y1,20,-30)
    x4b,y4b = calculate_new_point(x4,y4,20,-30)
    x2b,y2b = calculate_new_point(x2,y2,20,-30)
    x3b,y3b = calculate_new_point(x3,y3,20,-30)
    x11b,y11b = calculate_new_point(x11,y11,20,-30)
    x41b,y41b = calculate_new_point(x41,y41,20,-30)
    x21b,y21b = calculate_new_point(x21,y21,20,-30)
    x31b,y31b = calculate_new_point(x31,y31,20,-30)
    contur2 = [(-80,35),(-80+15,35),(-80+15,35+4),(-80+45,35+4),(-80+45,35),(80-45,35),(80-45,35+4),(80-15,35+4),(80-15,35)] +  get_circle_points(center=(-bot_cx, bot_cy),radius=radius,begin_angle=90,end_angle=-30) + [(x1b,y1b),(x2b,y2b),(x3b,y3b),(x4b,y4b),(x11b,y11b),(x21b,y21b),(x31b,y31b),(x41b,y41b)] + get_circle_points(center=(c3x,c3y),radius=55,begin_angle=-30,end_angle=-90) + [(x12,y12-20),(x22,y22-20),(x32,y32-20),(x42,y42-20)] + [(-x12,y12-20),(-x22,y22-20),(-x32,y32-20),(-x42,y42-20)][::-1] + get_circle_points(center=(c2x,c2y),radius=55,begin_angle=270,end_angle=210) + [(-x1b,y1b),(-x2b,y2b),(-x3b,y3b),(-x4b,y4b),(-x11b,y11b),(-x21b,y21b),(-x31b,y31b),(-x41b,y41b)][::-1] + get_circle_points(center=(bot_cx, bot_cy),radius=radius,begin_angle=210,end_angle=90)
    data_base.add_coordinates(f"contur2", contur2)
    data_base.add_polyline(f"contur2","top",1, False, True, False)
    print(calculate_polyline_length([get_circle_points(center=(-bot_cx, bot_cy),radius=radius,begin_angle=90,end_angle=-30)[-1],(x1b,y1b)]))


    redraw()

def get_schip_points(center,width,height):
    return [(center[0]-width/2,center[1]),(center[0]-width/2,center[1]+height),(center[0]+width/2,center[1]+height),(center[0]+width/2,center[1])]
def get_bok_schip_points(center,width,height):
    return [(center[0],center[1]+ width/2),(center[0]+height,center[1]+ width/2),(center[0]+height,center[1]- width/2),(center[0],center[1]- width/2)]

def test11():
    dpg.add_button(label="CNC",parent='butonss',tag="CNC",callback=active_but)
    ang = 15
    p = rotate_points(points=(25,220),center=(0,220),angle_degrees=ang)
    p1 = rotate_points(points=(-25,220),center=(0,220),angle_degrees=-ang)
    angle_rad = math.atan2(40 - p1[1], -p1[0]-90)
    ang_degr = math.degrees(angle_rad)
    a = calculate_new_point2(p1,20,ang_degr)
    b = calculate_new_point2(a,6,ang_degr+90)
    c = calculate_new_point2(a,20,ang_degr)
    d = calculate_new_point2(c,6,ang_degr+90)


    a1 = calculate_new_point2(c,20,ang_degr)
    b1 = calculate_new_point2(a1,6,ang_degr+90)
    c1 = calculate_new_point2(a1,20,ang_degr)
    d1 = calculate_new_point2(c1,6,ang_degr+90)

    a2 = calculate_new_point2(c1,20,ang_degr)
    b2 = calculate_new_point2(a2,6,ang_degr+90)
    c2 = calculate_new_point2(a2,20,ang_degr)
    d2 = calculate_new_point2(c2,6,ang_degr+90)

    data_base.add_coordinates(f"ccn", get_circle_points(center=(0,100),radius=6))
    data_base.add_polyline(f"ccn","CNC",0, False, True, False)    
    data_base.add_coordinates(f"ccn1", get_circle_points(center=(0,220),radius=6))
    data_base.add_polyline(f"ccn1","CNC",0, False, True, False)    
    data_base.add_coordinates(f"cntr", get_circle_points(center=(0,220),radius=25,begin_angle=ang,end_angle=180-ang) + [ a,b,d,c,a1,b1,d1,c1,a2,b2,d2,c2,(-90,40),(-150,40),(-150,-40),(150,-40),(150,40),(90,40),p])
    data_base.add_polyline(f"cntr","CNC",0, False, True, False)    
    
    data_base.add_coordinates(f"a1", get_rectangle_points_from_center(width=20,height=6,center=(0,-3)))
    data_base.add_polyline(f"a1","CNC",0, False, True, False)    
    data_base.add_coordinates(f"a2", get_rectangle_points_from_center(width=20,height=6,center=(40,-3)))
    data_base.add_polyline(f"a2","CNC",0, False, True, False)   
    data_base.add_coordinates(f"a3", get_rectangle_points_from_center(width=20,height=6,center=(80,-3)))
    data_base.add_polyline(f"a3","CNC",0, False, True, False) 
    data_base.add_coordinates(f"a4", get_rectangle_points_from_center(width=20,height=6,center=(120,-3)))
    data_base.add_polyline(f"a4","CNC",0, False, True, False) 
    data_base.add_coordinates(f"a5", get_rectangle_points_from_center(width=20,height=6,center=(-40,-3)))
    data_base.add_polyline(f"a5","CNC",0, False, True, False)   
    data_base.add_coordinates(f"a6", get_rectangle_points_from_center(width=20,height=6,center=(-80,-3)))
    data_base.add_polyline(f"a6","CNC",0, False, True, False) 
    data_base.add_coordinates(f"a7", get_rectangle_points_from_center(width=20,height=6,center=(-120,-3)))
    data_base.add_polyline(f"a7","CNC",0, False, True, False) 


    data_base.add_coordinates(f"1a1", get_rectangle_points_from_center(width=20,height=6,center=(0,-33)))
    data_base.add_polyline(f"1a1","CNC",0, False, True, False)    
    data_base.add_coordinates(f"1a2", get_rectangle_points_from_center(width=20,height=6,center=(40,-33)))
    data_base.add_polyline(f"1a2","CNC",0, False, True, False)   
    data_base.add_coordinates(f"1a3", get_rectangle_points_from_center(width=20,height=6,center=(80,-33)))
    data_base.add_polyline(f"1a3","CNC",0, False, True, False) 
    data_base.add_coordinates(f"1a4", get_rectangle_points_from_center(width=20,height=6,center=(120,-33)))
    data_base.add_polyline(f"1a4","CNC",0, False, True, False) 
    data_base.add_coordinates(f"1a5", get_rectangle_points_from_center(width=20,height=6,center=(-40,-33)))
    data_base.add_polyline(f"1a5","CNC",0, False, True, False)   
    data_base.add_coordinates(f"1a6", get_rectangle_points_from_center(width=20,height=6,center=(-80,-33)))
    data_base.add_polyline(f"1a6","CNC",0, False, True, False) 
    data_base.add_coordinates(f"1a7", get_rectangle_points_from_center(width=20,height=6,center=(-120,-33)))
    data_base.add_polyline(f"1a7","CNC",0, False, True, False) 


    data_base.add_coordinates(f"2a1", get_rectangle_points_from_center(width=6,height=20,center=(-141,20)))
    data_base.add_polyline(f"2a1","CNC",0, False, True, False)    
    data_base.add_coordinates(f"2a2", get_rectangle_points_from_center(width=6,height=20,center=(-141,-20)))
    data_base.add_polyline(f"2a2","CNC",0, False, True, False)
    data_base.add_coordinates(f"2a3", get_rectangle_points_from_center(width=6,height=20,center=(141,20)))
    data_base.add_polyline(f"2a3","CNC",0, False, True, False)    
    data_base.add_coordinates(f"2a4", get_rectangle_points_from_center(width=6,height=20,center=(141,-20)))
    data_base.add_polyline(f"2a4","CNC",0, False, True, False)

    data_base.add_coordinates(f"1nema17", get_circle_points(center=(0+15.5,160-15.5),radius=1.5))
    data_base.add_polyline(f"1nema17","CNC",0, False, True, False) 
    data_base.add_coordinates(f"2nema17", get_circle_points(center=(0+15.5,160+15.5),radius=1.5))
    data_base.add_polyline(f"2nema17","CNC",0, False, True, False)
    data_base.add_coordinates(f"3nema17", get_circle_points(center=(0-15.5,160+15.5),radius=1.5))
    data_base.add_polyline(f"3nema17","CNC",0, False, True, False)
    data_base.add_coordinates(f"4nema17", get_circle_points(center=(0-15.5,160-15.5),radius=1.5))
    data_base.add_polyline(f"4nema17","CNC",0, False, True, False)
    data_base.add_coordinates(f"5nema17", get_circle_points(center=(0,160),radius=10))
    data_base.add_polyline(f"5nema17","CNC",0, False, True, False)





    data_base.add_coordinates(f"napr", get_circle_points(center=(32,60),radius=6))
    data_base.add_polyline(f"napr","ctena1",0, False, True, False)
    data_base.add_coordinates(f"napr1", get_circle_points(center=(232,60),radius=6))
    data_base.add_polyline(f"napr1","ctena1",0, False, True, False)

    data_base.add_coordinates(f"1nema172", get_circle_points(center=(132+15.5,60-15.5),radius=1.5))
    data_base.add_polyline(f"1nema172","ctena1",0, False, True, False) 
    data_base.add_coordinates(f"2nema172", get_circle_points(center=(132+15.5,60+15.5),radius=1.5))
    data_base.add_polyline(f"2nema172","ctena1",0, False, True, False)
    data_base.add_coordinates(f"3nema172", get_circle_points(center=(132-15.5,60+15.5),radius=1.5))
    data_base.add_polyline(f"3nema172","ctena1",0, False, True, False)
    data_base.add_coordinates(f"4nema172", get_circle_points(center=(132-15.5,60-15.5),radius=1.5))
    data_base.add_polyline(f"4nema172","ctena1",0, False, True, False)
    data_base.add_coordinates(f"5nema172", get_circle_points(center=(132,60),radius=10))
    data_base.add_polyline(f"5nema172","ctena1",0, False, True, False)

    dpg.add_button(label="ctena1",parent='butonss',tag="ctena1",callback=active_but)
    data_base.add_coordinates(f"ctena11", [(0,10),(12,10),(12,0),(252,0),(252,10),(264,10),(264,30),(252,30),(252,50),(264,50),(264,70),(252,70),(252,80),(12,80),(12,70),(0,70),(0,50),(12,50),(12,30),(0,30),(0,10)])
    data_base.add_polyline(f"ctena11","ctena1",0, False, True, False)

    data_base.add_coordinates(f"ct11", get_rectangle_points_from_center(width=20,height=6,center=(32,7)))
    data_base.add_polyline(f"ct11","ctena1",0, False, True, False)
    data_base.add_coordinates(f"ct12", get_rectangle_points_from_center(width=20,height=6,center=(72,7)))
    data_base.add_polyline(f"ct12","ctena1",0, False, True, False)
    data_base.add_coordinates(f"ct13", get_rectangle_points_from_center(width=20,height=6,center=(112,7)))
    data_base.add_polyline(f"ct13","ctena1",0, False, True, False)
    data_base.add_coordinates(f"ct14", get_rectangle_points_from_center(width=20,height=6,center=(152,7)))
    data_base.add_polyline(f"ct14","ctena1",0, False, True, False)
    data_base.add_coordinates(f"ct15", get_rectangle_points_from_center(width=20,height=6,center=(192,7)))
    data_base.add_polyline(f"ct15","ctena1",0, False, True, False)
    data_base.add_coordinates(f"ct16", get_rectangle_points_from_center(width=20,height=6,center=(232,7)))
    data_base.add_polyline(f"ct16","ctena1",0, False, True, False)


    data_base.add_coordinates(f"2ct11", get_rectangle_points_from_center(width=20,height=6,center=(32,37)))
    data_base.add_polyline(f"2ct11","ctena1",0, False, True, False)
    data_base.add_coordinates(f"2ct12", get_rectangle_points_from_center(width=20,height=6,center=(72,37)))
    data_base.add_polyline(f"2ct12","ctena1",0, False, True, False)
    data_base.add_coordinates(f"2ct13", get_rectangle_points_from_center(width=20,height=6,center=(112,37)))
    data_base.add_polyline(f"2ct13","ctena1",0, False, True, False)
    data_base.add_coordinates(f"2ct14", get_rectangle_points_from_center(width=20,height=6,center=(152,37)))
    data_base.add_polyline(f"2ct14","ctena1",0, False, True, False)
    data_base.add_coordinates(f"2ct15", get_rectangle_points_from_center(width=20,height=6,center=(192,37)))
    data_base.add_polyline(f"2ct15","ctena1",0, False, True, False)
    data_base.add_coordinates(f"2ct16", get_rectangle_points_from_center(width=20,height=6,center=(232,37)))
    data_base.add_polyline(f"2ct16","ctena1",0, False, True, False)



    dpg.add_button(label="ctena3",parent='butonss',tag="ctena3",callback=active_but)
    data_base.add_coordinates(f"ctena31", [(0,10),(12,10),(12,0),(252,0),(252,10),(264,10),(264,30),(252,30),(252,50),(264,50),(264,70),(252,70),(252,90),(264,90),(264,110),(252,110),(252,120),(12,120),(12,110),(0,110),(0,90),(12,90),(12,70),(0,70),(0,50),(12,50),(12,30),(0,30),(0,10)])
    data_base.add_polyline(f"ctena31","ctena3",0, False, True, False)

    dpg.add_button(label="base1",parent='butonss',tag="base1",callback=active_but)
    data_base.add_coordinates(f"base11",[(-138,0)] + get_schip_points(center=(-120,0),width=20,height=12) + get_schip_points(center=(-80,0),width=20,height=12)+get_schip_points(center=(-40,0),width=20,height=12)+ get_schip_points(center=(0,0),width=20,height=12)+ get_schip_points(center=(40,0),width=20,height=12)+ get_schip_points(center=(80,0),width=20,height=12)+ get_schip_points(center=(120,0),width=20,height=12) + [(138,0)] + get_bok_schip_points(center=(138,-20),width=20,height=6)+ get_bok_schip_points(center=(138,-60),width=20,height=6)+ get_bok_schip_points(center=(138,-100),width=20,height=6)+ get_bok_schip_points(center=(138,-140),width=20,height=6)+ get_bok_schip_points(center=(138,-180),width=20,height=6)+ get_bok_schip_points(center=(138,-220),width=20,height=6) + [(138,-240)] + get_schip_points(center=(120,-240),width=-20,height=-12)+ get_schip_points(center=(80,-240),width=-20,height=-12)+ get_schip_points(center=(40,-240),width=-20,height=-12)+ get_schip_points(center=(0,-240),width=-20,height=-12)+ get_schip_points(center=(-40,-240),width=-20,height=-12)+ get_schip_points(center=(-80,-240),width=-20,height=-12)+ get_schip_points(center=(-120,-240),width=-20,height=-12) + [(-138,-240)] + get_bok_schip_points(center=(-138,-220),width=-20,height=-6)+get_bok_schip_points(center=(-138,-180),width=-20,height=-6)+get_bok_schip_points(center=(-138,-140),width=-20,height=-6)+get_bok_schip_points(center=(-138,-100),width=-20,height=-6)+get_bok_schip_points(center=(-138,-60),width=-20,height=-6)+get_bok_schip_points(center=(-138,-20),width=-20,height=-6) + [(-138,0)])
    data_base.add_polyline(f"base11","base1",0, False, True, False)

    
    redraw()




def test5():
    dpg.add_button(label="bottom",parent='butonss',tag="bottom",callback=active_but)
    dpg.add_button(label="top",parent='butonss',tag="top",callback=active_but)
    dpg.add_button(label="panel",parent='butonss',tag="panel",callback=active_but)
    dpg.add_button(label="panel2",parent='butonss',tag="panel2",callback=active_but)
    dpg.add_button(label="panel3",parent='butonss',tag="panel3",callback=active_but)
    dpg.add_button(label="panel4",parent='butonss',tag="panel4",callback=active_but)
    big_c =  get_circle_points(center=(0,0),radius=35,begin_angle=0,end_angle=360)
    r1 =  get_circle_points(center=(-80,0),radius=20,begin_angle=0,end_angle=360)
    r4 =  get_circle_points(center=(80,0),radius=20,begin_angle=0,end_angle=360)
    data_base.add_coordinates(f"bc", big_c)
    data_base.add_polyline(f"bc","top",0, False, True, False)
    data_base.add_coordinates(f"r1", r1)
    data_base.add_polyline(f"r1","top",0, False, True, False)
    data_base.add_coordinates(f"r4", r4)
    data_base.add_polyline(f"r4","top",0, False, True, False)

    c2x,c2y = calculate_new_point(0,0,80,-120)
    r2 =  get_circle_points(center=(c2x,c2y),radius=20,begin_angle=0,end_angle=360)
    data_base.add_coordinates(f"r2", r2)
    data_base.add_polyline(f"r2","top",0, False, True, False)

    c3x,c3y = calculate_new_point(0,0,80,-60)
    r3 =  get_circle_points(center=(c3x,c3y),radius=20,begin_angle=0,end_angle=360)
    data_base.add_coordinates(f"r3", r3)
    data_base.add_polyline(f"r3","top",0, False, True, False)

    data_base.add_coordinates(f"sw1", get_circle_points(center=(-80,0-3.5),radius=2.5,begin_angle=0,end_angle=360))
    data_base.add_polyline(f"sw1","top",0, False, True, False)
    data_base.add_polyline(f"kn1","top",0, False, True, False)
    data_base.add_coordinates("kn1", get_rectangle_points_from_center(width=12.9,height=5.6,center=(-80,0+4)))

    data_base.add_coordinates(f"sw2", get_circle_points(center=(c2x,c2y-3.5),radius=2.5,begin_angle=0,end_angle=360))
    data_base.add_polyline(f"sw2","top",0, False, True, False)
    data_base.add_polyline(f"kn2","top",0, False, True, False)
    data_base.add_coordinates("kn2", get_rectangle_points_from_center(width=12.9,height=5.6,center=(c2x,c2y+4)))

    data_base.add_coordinates(f"sw3", get_circle_points(center=(c3x,c3y-3.5),radius=2.5,begin_angle=0,end_angle=360))
    data_base.add_polyline(f"sw3","top",0, False, True, False)
    data_base.add_polyline(f"kn3","top",0, False, True, False)
    data_base.add_coordinates("kn3", get_rectangle_points_from_center(width=12.9,height=5.6,center=(c3x,c3y+4)))

    data_base.add_coordinates(f"sw4", get_circle_points(center=(80,0-3.5),radius=2.5,begin_angle=0,end_angle=360))
    data_base.add_polyline(f"sw4","top",0, False, True, False)
    data_base.add_polyline(f"kn4","top",0, False, True, False)
    data_base.add_coordinates("kn4", get_rectangle_points_from_center(width=12.9,height=5.6,center=(80,0+4)))




    x,y = find_intersection((105,0),(105,-30),calculate_new_point(80,0,25,-30),calculate_new_point(c3x,c3y,25,-30))
    x2,y2 = find_intersection((c2x,c2y-25),(c3x,c3y-25),calculate_new_point(80,0,25,-30),calculate_new_point(c3x,c3y,25,-30))    
    x3,y3 = find_intersection((c2x,c2y-45),(c3x,c3y-45),calculate_new_point(80,0,25,-30),calculate_new_point(c3x,c3y,25,-30))  

    p1 = calculate_new_point(80,0,25,-30)
    p2 = calculate_new_point(80,0,30,-30)
    p3 = calculate_new_point2(calculate_new_point(80,0,30,-30),30,-120)
    p4 = calculate_new_point2(calculate_new_point2(calculate_new_point(80,0,29,-30),30,-120),4,150)


    p5 = calculate_new_point(c3x,c3y,25,-30)
    p6 = calculate_new_point(c3x,c3y,30,-30)
    p7 = calculate_new_point2(p6,30,60)
    p8 = calculate_new_point2(p5,30,60)


    contur = [(-105,40),(-100+30,40),(-100+30,40+4),(-100+60,40+4),(-100+60,40),(100-60,40),(100-60,40+4),(100-30,40+4),(100-30,40),(105,40),(105,30),(109,30),(109,0),(105,0),(x,y),p1,p2,p3,p4,p8,p7,p6,p5,(x2,y2),(x2-20,y2),(x2-20,y2-4),(x2-40,y2-4),(x2-40,y2),(-x2+40,y2),(-x2+40,y2-4),(-x2+20,y2-4),(-x2+20,y2),(-x2,y2) , (-p5[0],p5[1]), (-p6[0],p6[1]), (-p7[0],p7[1]), (-p8[0],p8[1]), (-p4[0],p4[1]), (-p3[0],p3[1]), (-p2[0],p2[1]), (-p1[0],p1[1]),(-x,y), (-105,0),(-109,0),(-109,30),(-105,30) ,(-105,40)] 

    data_base.add_coordinates(f"contur", contur)
    data_base.add_polyline(f"contur","top",0, False, True, False)

    contur2 = [(-105,40),(-100+30,40),(-100+30,40+4),(-100+60,40+4),(-100+60,40),(100-60,40),(100-60,40+4),(100-30,40+4),(100-30,40),(105,40),(105,30),(109,30),(109,0),(105,0),(x,y),p1,p2,p3,p4,p8,p7,p6,p5,(x3,y3),(x2-20,y3),(x2-20,y3-4),(x2-40,y3-4),(x2-40,y3),(-x2+40,y3),(-x2+40,y3-4),(-x2+20,y3-4),(-x2+20,y3),(-x3,y3) , (-p5[0],p5[1]), (-p6[0],p6[1]), (-p7[0],p7[1]), (-p8[0],p8[1]), (-p4[0],p4[1]), (-p3[0],p3[1]), (-p2[0],p2[1]), (-p1[0],p1[1]),(-x,y), (-105,0),(-109,0),(-109,30),(-105,30) ,(-105,40)] 

    data_base.add_coordinates(f"contur2", contur2)
    data_base.add_polyline(f"contur2","bottom",1, False, True, False)


    contur3 = [(-x3-4,0),(-x2+20,0),(-x2+20,4),(-x2+40,4),(-x2+40,0),(x2-40,0),(x2-40,4),(x2-20,4),(x2-20,0),(x3+4,0),(x2+4,40.31),(x2-20,40.31),(x2-20,40.31-4) ,(x2-40,40.31-4),(x2-40,40.31),(-x2+40,40.31) ,(-x2+40,40.31-4) ,(-x2+20,40.31-4),(-x2+20,40.31) ,(-x2-4,40.31),(-x3-4,0)] 

    data_base.add_coordinates(f"contur3", contur3)
    data_base.add_polyline(f"contur3","panel",1, False, True, False)
    
    data_base.add_polyline(f"potent1","panel",1, False, True, False)
    data_base.add_coordinates(f"potent1", get_circle_points(center=(24,20),radius=3.5,begin_angle=0,end_angle=360))
   
    data_base.add_polyline(f"oled1","panel",1, False, True, False)
    data_base.add_coordinates(f"oled1", [(-36.5,11.5),(-36.5,28.5),(-11.5,28.5),(-11.5,11.5),(-36.5,11.5)])

    
    
    contur4 = [(4,0),(39,0),(39,4),(69,4),(69,0),(149,0),(149,4),(179,4),(179,0),(214,0),(214,10),(218,10),(218,25) ,(214,25),(214,35),(179,35) ,(179,31) ,(149,31),(149,35) ,(69,35),(69,31),(39,31),(39,35),(4,35),(4,25),(0,25),(0,10),(4,10),(4,0)] 

    data_base.add_coordinates(f"contur4", contur4)
    data_base.add_polyline(f"contur4","panel2",1, False, True, False)
    data_base.add_coordinates(f"type-c", get_circle_points(center=(24,8),radius=3,begin_angle=90,end_angle=270) + get_circle_points(center=(30,8),radius=3,begin_angle=-90,end_angle=90) + [(24,11)])
    data_base.add_polyline(f"type-c","panel2",1, False, True, False)
    data_base.add_coordinates(f"onoff",get_rectangle_points_from_center(width=8,height=12,center=(200,17.5)))
    data_base.add_polyline(f"onoff","panel2",1, False, True, False)

    contur5 = [(0,10),(0,0),(10,0),(10,4),(40,4),(40,0),(46.7,0),(46.7,10),(42.7,10),(42.7,25),(46.7,25),(46.7,35),(40,35) ,(40,31),(10,31),(10,35) ,(0,35) ,(0,25),(4,25) ,(4,10),(0,10)] 

    data_base.add_coordinates(f"contur5", contur5)
    data_base.add_polyline(f"contur5","panel3",1, False, True, False)

    contur6 = [(-4,10),(-4,0),(6.7,0),(6.7,4),(36.7,4),(36.7,0),(56.7,0),(56.7,4),(86.7,4),(86.7,0),(124.3,0),(101.1,35),(86.7,35) ,(86.7,31),(56.7,31),(56.7,35) ,(36.7,35) ,(36.7,31),(6.7,31) ,(6.7,35),(-4,35),(-4,25),(0,25),(0,10),(-4,10)] 

    data_base.add_coordinates(f"contur6", contur6)
    data_base.add_polyline(f"contur6","panel4",1, False, True, False)


    redraw()


def testvlad1():

    dpg.add_button(label="vlad1",parent='butonss',tag="vlad1",callback=active_but)
    dpg.add_button(label="vlad2",parent='butonss',tag="vlad2",callback=active_but)
    dpg.add_button(label="vlad3",parent='butonss',tag="vlad3",callback=active_but)
    
    data_base.add_coordinates(f"vlad11", [(4.02, 0.0), (19.02, 0.0), (19.03, 0.0), (19.04, 0.01), (19.04, 0.02), (19.04, 4.0), (27.0, 4.0), (27.0, 0.02), (27.0, 0.01), (27.01, 0.0), (27.02, 0.0), (35.02, 0.0), (35.03, 0.0), (35.04, 0.01), (35.04, 0.02), (35.04, 4.0), (43.0, 4.0), (43.0, 0.02), (43.0, 0.01), (43.01, 0.0), (43.02, 0.0), (51.02, 0.0), (51.03, 0.0), (51.04, 0.01), (51.04, 0.02), (51.04, 4.0), (59.0, 4.0), (59.0, 0.02), (59.0, 0.01), (59.01, 0.0), (59.02, 0.0), (67.02, 0.0), (67.03, 0.0), (67.04, 0.01), (67.04, 0.02), (67.04, 4.0), (75.0, 4.0), (75.0, 0.02), (75.0, 0.01), (75.01, 0.0), (75.02, 0.0), (83.02, 0.0), (83.03, 0.0), (83.04, 0.01), (83.04, 0.02), (83.04, 4.0), (91.0, 4.0), (91.0, 0.02), (91.0, 0.01), (91.01, 0.0), (91.02, 0.0), (99.02, 0.0), (99.03, 0.0), (99.04, 0.01), (99.04, 0.02), (99.04, 4.0), (107.0, 4.0), (107.0, 0.02), (107.0, 0.01), (107.01, 0.0), (107.02, 0.0), (115.02, 0.0), (115.03, 0.0), (115.04, 0.01), (115.04, 0.02), (115.04, 4.0), (123.0, 4.0), (123.0, 0.02), (123.0, 0.01), (123.01, 0.0), (123.02, 0.0), (131.02, 0.0), (131.03, 0.0), (131.04, 0.01), (131.04, 0.02), (131.04, 4.0), (139.0, 4.0), (139.0, 0.02), (139.0, 0.01), (139.01, 0.0), (139.02, 0.0), (147.02, 0.0), (147.03, 0.0), (147.04, 0.01), (147.04, 0.02), (147.04, 4.0), (155.0, 4.0), (155.0, 0.02), (155.0, 0.01), (155.01, 0.0), (155.02, 0.0), (163.02, 0.0), (163.03, 0.0), (163.04, 0.01), (163.04, 0.02), (163.04, 4.0), (171.0, 4.0), (171.0, 0.02), (171.0, 0.01), (171.01, 0.0), (171.02, 0.0), (186.02, 0.0), (186.03, 0.0), (186.04, 0.01), (186.04, 0.02), (186.04, 8.5), (190.02, 8.5), (190.03, 8.5), (190.04, 8.51), (190.04, 8.52), (190.04, 16.52), (190.04, 16.53), (190.03, 16.54), (190.02, 16.54), (186.04, 16.54), (186.04, 24.5), (190.02, 24.5), (190.03, 24.5), (190.04, 24.51), (190.04, 24.52), (190.04, 32.52), (190.04, 32.53), (190.03, 32.54), (190.02, 32.54), (186.04, 32.54), (186.04, 40.5), (190.02, 40.5), (190.03, 40.5), (190.04, 40.51), (190.04, 40.52), (190.04, 48.52), (190.04, 48.53), (190.03, 48.54), (190.02, 48.54), (186.04, 48.54), (186.04, 56.5), (190.02, 56.5), (190.03, 56.5), (190.04, 56.51), (190.04, 56.52), (190.04, 64.52), (190.04, 64.53), (190.03, 64.54), (190.02, 64.54), (186.04, 64.54), (186.04, 73.02), (186.04, 73.03), (186.03, 73.04), (186.02, 73.04), (4.02, 73.04), (4.01, 73.04), (4.0, 73.03), (4.0, 73.02), (4.0, 64.54), (0.02, 64.54), (0.01, 64.54), (0.0, 64.53), (0.0, 64.52), (0.0, 56.52), (0.0, 56.51), (0.01, 56.5), (0.02, 56.5), (4.0, 56.5), (4.0, 48.54), (0.02, 48.54), (0.01, 48.54), (0.0, 48.53), (0.0, 48.52), (0.0, 40.52), (0.0, 40.51), (0.01, 40.5), (0.02, 40.5), (4.0, 40.5), (4.0, 32.54), (0.02, 32.54), (0.01, 32.54), (0.0, 32.53), (0.0, 32.52), (0.0, 24.52), (0.0, 24.51), (0.01, 24.5), (0.02, 24.5), (4.0, 24.5), (4.0, 16.54), (0.02, 16.54), (0.01, 16.54), (0.0, 16.53), (0.0, 16.52), (0.0, 8.52), (0.0, 8.51), (0.01, 8.5), (0.02, 8.5), (4.0, 8.5), (4.0, 0.02), (4.0, 0.01), (4.01, 0.0), (4.02, 0.0)])
    data_base.add_polyline(f"vlad11","vlad1",0, False, True, False)
    data_base.add_coordinates(f"vlad12", get_circle_points(center=(46+15+3,70.04),radius=3,begin_angle=90,end_angle=180) + get_circle_points(center=(46,70.04),radius=15,begin_angle=360,end_angle=180) + get_circle_points(center=(46-15-3,70.04),radius=3,begin_angle=0,end_angle=90)+[(0.01, 73.04), (0.0, 73.03), (0.0, 73.02), (0.0, 64.52), (0.0, 64.51), (0.01, 64.5), (0.02, 64.5), (4.0, 64.5), (4.0, 56.54), (0.02, 56.54), (0.01, 56.54), (0.0, 56.53), (0.0, 56.52), (0.0, 48.52), (0.0, 48.51), (0.01, 48.5), (0.02, 48.5), (4.0, 48.5), (4.0, 40.54), (0.02, 40.54), (0.01, 40.54), (0.0, 40.53), (0.0, 40.52), (0.0, 32.52), (0.0, 32.51), (0.01, 32.5), (0.02, 32.5), (4.0, 32.5), (4.0, 24.54), (0.02, 24.54), (0.01, 24.54), (0.0, 24.53), (0.0, 24.52), (0.0, 16.52), (0.0, 16.51), (0.01, 16.5), (0.02, 16.5), (4.0, 16.5), (4.0, 8.54), (0.02, 8.54), (0.01, 8.54), (0.0, 8.53), (0.0, 8.52), (0.0, 0.02), (0.0, 0.01), (0.01, 0.0), (0.02, 0.0),(4.02, 0.0), (18.02, 0.0), (18.03, 0.0), (18.04, 0.01), (18.04, 0.02), (18.04, 4.0), (26.0, 4.0), (26.0, 0.02), (26.0, 0.01), (26.01, 0.0), (26.02, 0.0), (34.02, 0.0), (34.03, 0.0), (34.04, 0.01), (34.04, 0.02), (34.04, 4.0), (42.0, 4.0), (42.0, 0.02), (42.0, 0.01), (42.01, 0.0), (42.02, 0.0), (50.02, 0.0), (50.03, 0.0), (50.04, 0.01), (50.04, 0.02), (50.04, 4.0), (58.0, 4.0), (58.0, 0.02), (58.0, 0.01), (58.01, 0.0), (58.02, 0.0), (66.02, 0.0), (66.03, 0.0), (66.04, 0.01), (66.04, 0.02), (66.04, 4.0), (74.0, 4.0), (74.0, 0.02), (74.0, 0.01), (74.01, 0.0), (74.02, 0.0), (88.02, 0.0), (92.02, 0.0), (92.03, 0.0), (92.04, 0.01), (92.04, 0.02), (92.04, 8.52), (92.04, 8.53), (92.03, 8.54), (92.02, 8.54), (88.04, 8.54), (88.04, 16.5), (92.02, 16.5), (92.03, 16.5), (92.04, 16.51), (92.04, 16.52), (92.04, 24.52), (92.04, 24.53), (92.03, 24.54), (92.02, 24.54), (88.04, 24.54), (88.04, 32.5), (92.02, 32.5), (92.03, 32.5), (92.04, 32.51), (92.04, 32.52), (92.04, 40.52), (92.04, 40.53), (92.03, 40.54), (92.02, 40.54), (88.04, 40.54), (88.04, 48.5), (92.02, 48.5), (92.03, 48.5), (92.04, 48.51), (92.04, 48.52), (92.04, 56.52), (92.04, 56.53), (92.03, 56.54), (92.02, 56.54), (88.04, 56.54), (88.04, 64.5), (92.02, 64.5), (92.03, 64.5), (92.04, 64.51), (92.04, 64.52), (92.04, 73.02), (92.04, 73.03), (92.03, 73.04), (64, 73.04)])
    data_base.add_polyline(f"vlad12","vlad2",0, False, True, False)
    data_base.add_coordinates(f"vlad13",[(186.04, 88.02), (186.04, 74.04), (190.02, 74.04), (190.03, 74.04), (190.04, 74.03), (190.04, 74.02), (190.04, 66.02), (190.04, 66.01), (190.03, 66.0), (190.02, 66.0), (186.04, 66.0), (186.04, 58.04), (190.02, 58.04), (190.03, 58.04), (190.04, 58.03), (190.04, 58.02), (190.04, 50.02), (190.04, 50.01), (190.03, 50.0), (190.02, 50.0), (186.04, 50.0), (186.04, 42.04), (190.02, 42.04), (190.03, 42.04), (190.04, 42.03), (190.04, 42.02), (190.04, 34.02), (190.04, 34.01), (190.03, 34.0), (190.02, 34.0), (186.04, 34.0), (186.04, 26.04), (190.02, 26.04), (190.03, 26.04), (190.04, 26.03), (190.04, 26.02), (190.04, 18.02), (190.04, 18.01), (190.03, 18.0), (190.02, 18.0), (186.04, 18.0), (186.04, 4.02), (186.04, 4.01), (186.03, 4.0), (186.02, 4.0), (171.04, 4.0), (171.04, 0.02), (171.04, 0.01), (171.03, 0.0), (171.02, 0.0), (163.02, 0.0), (163.01, 0.0), (163.0, 0.01), (163.0, 0.02), (163.0, 4.0), (155.04, 4.0), (155.04, 0.02), (155.04, 0.01), (155.03, 0.0), (155.02, 0.0), (147.02, 0.0), (147.01, 0.0), (147.0, 0.01), (147.0, 0.02), (147.0, 4.0), (139.04, 4.0), (139.04, 0.02), (139.04, 0.01), (139.03, 0.0), (139.02, 0.0), (131.02, 0.0), (131.01, 0.0), (131.0, 0.01), (131.0, 0.02), (131.0, 4.0), (123.04, 4.0), (123.04, 0.02), (123.04, 0.01), (123.03, 0.0), (123.02, 0.0), (115.02, 0.0), (115.01, 0.0), (115.0, 0.01), (115.0, 0.02), (115.0, 4.0), (107.04, 4.0), (107.04, 0.02), (107.04, 0.01), (107.03, 0.0), (107.02, 0.0), (99.02, 0.0), (99.01, 0.0), (99.0, 0.01), (99.0, 0.02), (99.0, 4.0), (91.04, 4.0), (91.04, 0.02), (91.04, 0.01), (91.03, 0.0), (91.02, 0.0), (83.02, 0.0), (83.01, 0.0), (83.0, 0.01), (83.0, 0.02), (83.0, 4.0), (75.04, 4.0), (75.04, 0.02), (75.04, 0.01), (75.03, 0.0), (75.02, 0.0), (67.02, 0.0), (67.01, 0.0), (67.0, 0.01), (67.0, 0.02), (67.0, 4.0), (59.04, 4.0), (59.04, 0.02), (59.04, 0.01), (59.03, 0.0), (59.02, 0.0), (51.02, 0.0), (51.01, 0.0), (51.0, 0.01), (51.0, 0.02), (51.0, 4.0), (43.04, 4.0), (43.04, 0.02), (43.04, 0.01), (43.03, 0.0), (43.02, 0.0), (35.02, 0.0), (35.01, 0.0), (35.0, 0.01), (35.0, 0.02), (35.0, 4.0), (27.04, 4.0), (27.04, 0.02), (27.04, 0.01), (27.03, 0.0), (27.02, 0.0), (19.02, 0.0), (19.01, 0.0), (19.0, 0.01), (19.0, 0.02), (19.0, 4.0), (4.02, 4.0), (4.01, 4.0), (4.0, 4.01), (4.0, 4.02), (4.0, 18.0), (0.02, 18.0), (0.01, 18.0), (0.0, 18.01), (0.0, 18.02), (0.0, 26.02), (0.0, 26.03), (0.01, 26.04), (0.02, 26.04), (4.0, 26.04), (4.0, 34.0), (0.02, 34.0), (0.01, 34.0), (0.0, 34.01), (0.0, 34.02), (0.0, 42.02), (0.0, 42.03), (0.01, 42.04), (0.02, 42.04), (4.0, 42.04), (4.0, 50.0), (0.02, 50.0), (0.01, 50.0), (0.0, 50.01), (0.0, 50.02), (0.0, 58.02), (0.0, 58.03), (0.01, 58.04), (0.02, 58.04), (4.0, 58.04), (4.0, 66.0), (0.02, 66.0), (0.01, 66.0), (0.0, 66.01), (0.0, 66.02), (0.0, 74.02), (0.0, 74.03), (0.01, 74.04), (0.02, 74.04), (4.0, 74.04), (4.0, 88.02), (4.0, 88.03), (4.01, 88.04), (4.02, 88.04), (19.0, 88.04), (19.0, 92.02), (19.0, 92.03), (19.01, 92.04), (19.02, 92.04), (27.02, 92.04), (27.03, 92.04), (27.04, 92.03), (27.04, 92.02), (27.04, 88.04), (35.0, 88.04), (35.0, 92.02), (35.0, 92.03), (35.01, 92.04), (35.02, 92.04), (43.02, 92.04), (43.03, 92.04), (43.04, 92.03), (43.04, 92.02), (43.04, 88.04), (51.0, 88.04), (51.0, 92.02), (51.0, 92.03), (51.01, 92.04), (51.02, 92.04), (59.02, 92.04), (59.03, 92.04), (59.04, 92.03), (59.04, 92.02), (59.04, 88.04), (67.0, 88.04), (67.0, 92.02), (67.0, 92.03), (67.01, 92.04), (67.02, 92.04), (75.02, 92.04), (75.03, 92.04), (75.04, 92.03), (75.04, 92.02), (75.04, 88.04), (83.0, 88.04), (83.0, 92.02), (83.0, 92.03), (83.01, 92.04), (83.02, 92.04), (91.02, 92.04), (91.03, 92.04), (91.04, 92.03), (91.04, 92.02), (91.04, 88.04), (99.0, 88.04), (99.0, 92.02), (99.0, 92.03), (99.01, 92.04), (99.02, 92.04), (107.02, 92.04), (107.03, 92.04), (107.04, 92.03), (107.04, 92.02), (107.04, 88.04), (115.0, 88.04), (115.0, 92.02), (115.0, 92.03), (115.01, 92.04), (115.02, 92.04), (123.02, 92.04), (123.03, 92.04), (123.04, 92.03), (123.04, 92.02), (123.04, 88.04), (131.0, 88.04), (131.0, 92.02), (131.0, 92.03), (131.01, 92.04), (131.02, 92.04), (139.02, 92.04), (139.03, 92.04), (139.04, 92.03), (139.04, 92.02), (139.04, 88.04), (147.0, 88.04), (147.0, 92.02), (147.0, 92.03), (147.01, 92.04), (147.02, 92.04), (155.02, 92.04), (155.03, 92.04), (155.04, 92.03), (155.04, 92.02), (155.04, 88.04), (163.0, 88.04), (163.0, 92.02), (163.0, 92.03), (163.01, 92.04), (163.02, 92.04), (171.02, 92.04), (171.03, 92.04), (171.04, 92.03), (171.04, 92.02), (171.04, 88.04), (186.02, 88.04), (186.03, 88.04), (186.04, 88.03), (186.04, 88.02)])
    data_base.add_polyline(f"vlad13","vlad3",0, False, True, False)
    
    data_base.add_coordinates(f"vlad122",get_circle_points(center=(46-15-3,70.04),radius=3,begin_angle=90,end_angle=0) + get_circle_points(center=(46,70.04),radius=15,begin_angle=180,end_angle=360) + get_circle_points(center=(46+15+3,70.04),radius=3,begin_angle=180,end_angle=90))

    # data_base.add_coordinates(f"vlad124",get_rectangle_points_from_left_bot(width=10,height=4,lb=(16,30)))
    # data_base.add_coordinates(f"vlad123",get_rectangle_points_from_left_bot(width=10,height=4,lb=(66,30)))

    data_base.add_polyline(f"vlad122","vlad2",0, False, True, False)
    data_base.add_polyline(f"vlad123","vlad2",0, False, True, False)
    redraw()


def test3():
    dpg.add_button(label="sekciya",parent='butonss',tag="sekciya",callback=active_but)
    data_base.add_coordinates(f"ss", [(0, 270)] + get_schip_points(center=(30,270),width=20,height=-6)+ get_schip_points(center=(90,270),width=60,height=-6)+ get_schip_points(center=(150,270),width=20,height=-6)+ get_schip_points(center=(210,270),width=60,height=-6)+ get_schip_points(center=(270,270),width=20,height=-6) + [(300,270),(300,6)] + get_schip_points(center=(210,6),width=-60,height=-6) + get_schip_points(center=(90,6),width=-60,height=-6) + [(0,6),(0,270)])
    data_base.add_polyline(f"ss","sekciya",0, False, True, False)


    data_base.add_coordinates(f"ss1", get_rectangle_points_from_center(width=20,height=6,center=(30,69)))
    data_base.add_coordinates(f"ss2", get_rectangle_points_from_center(width=20,height=6,center=(30,69+66)))
    data_base.add_coordinates(f"ss3", get_rectangle_points_from_center(width=20,height=6,center=(30,69+66+66)))

    data_base.add_coordinates(f"ss4", get_rectangle_points_from_center(width=20,height=6,center=(150,69)))
    data_base.add_coordinates(f"ss5", get_rectangle_points_from_center(width=20,height=6,center=(150,69+66)))
    data_base.add_coordinates(f"ss6", get_rectangle_points_from_center(width=20,height=6,center=(150,69+66+66)))

    data_base.add_coordinates(f"ss7", get_rectangle_points_from_center(width=20,height=6,center=(270,69)))
    data_base.add_coordinates(f"ss8", get_rectangle_points_from_center(width=20,height=6,center=(270,69+66)))
    data_base.add_coordinates(f"ss9", get_rectangle_points_from_center(width=20,height=6,center=(270,69+66+66)))
    data_base.add_polyline(f"ss1","sekciya",0, False, True, False)
    data_base.add_polyline(f"ss2","sekciya",0, False, True, False)
    data_base.add_polyline(f"ss3","sekciya",0, False, True, False)
    data_base.add_polyline(f"ss4","sekciya",0, False, True, False)
    data_base.add_polyline(f"ss5","sekciya",0, False, True, False)
    data_base.add_polyline(f"ss6","sekciya",0, False, True, False)
    data_base.add_polyline(f"ss7","sekciya",0, False, True, False)
    data_base.add_polyline(f"ss8","sekciya",0, False, True, False)
    data_base.add_polyline(f"ss9","sekciya",0, False, True, False)


    dpg.add_button(label="polka",parent='butonss',tag="polka",callback=active_but)
    data_base.add_coordinates(f"sss", [(0, 294)] + get_schip_points(center=(30,294),width=20,height=6)  + get_schip_points(center=(150,294),width=20,height=6)  + get_schip_points(center=(270,294),width=20,height=6)  + [(300,294),(300,6)] + get_schip_points(center=(270,6),width=-20,height=-6) + get_schip_points(center=(150,6),width=-20,height=-6)+ get_schip_points(center=(30,6),width=-20,height=-6) + [(0,6),(0,294)])
    data_base.add_polyline(f"sss","polka",0, False, True, False)

    redraw()

def testkoleso():
    # alph = 10
    # r = 138 * np.cos(math.radians(alph)) 
    # b =  138 * 2  * np.sin(math.radians(alph))
    # faska = 1.5
    dpg.add_button(label="koleso",parent='butonss',tag="koleso",callback=active_but)
    # dpg.add_button(label="v",parent='butonss',tag="v",callback=active_but)
    
    # data_base.add_coordinates(f"kol", get_circle_points(center=(0,0),radius=150))
    # data_base.add_polyline(f"kol","koleso",0, False, True, False)
    # data_base.add_coordinates(f"kol2", get_circle_points(center=(0,0),radius=10))
    # data_base.add_polyline(f"kol2","koleso",0, False, True, False)
    
    # data_base.add_coordinates(f"kol3", get_rectangle_points_from_center(width=b/2,height=6,center=(0,r+3)))
    # data_base.add_polyline(f"kol3","v",0, False, True, False)
    
    dpg.add_button(label="pol",parent='butonss',tag="pol",callback=active_but)
    # data_base.add_coordinates(f"з0", [(0,b/4+faska),(0+faska,b/4),(6,b/4),(6,0),(166,0),(166,b/4),(172-faska,b/4),(172,b/4+faska),(172,3*b/4-faska),(172-faska,3*b/4),(166,3*b/4),(166,b),(6,b),(6,3*b/4),(0+faska,3*b/4),(0,3*b/4-faska),(0,b/4+faska)])
    # data_base.add_polyline(f"з0","pol",0, False, True, False)

    data_base.add_coordinates(f"k1", [(159.51, 149.88), (159.38, 151.48), (159.0, 153.05), (158.37, 154.53), (157.51, 155.89), (156.44, 157.09), (155.19, 158.11), (153.8, 158.91), (152.3, 159.48), (150.72, 159.81), (149.11, 159.87), (147.51, 159.68), (145.97, 159.23), (144.51, 158.54), (143.19, 157.62), (142.03, 156.51), (141.06, 155.22), (140.31, 153.8), (139.8, 152.27), (139.55, 150.68), (139.55, 149.07), (139.8, 147.49), (140.31, 145.96), (141.06, 144.53), (142.03, 143.25), (143.19, 142.13), (144.51, 141.22), (145.97, 140.53), (147.51, 140.08), (149.11, 139.89), (150.72, 139.95), (152.3, 140.27), (153.8, 140.84), (155.19, 141.65), (156.44, 142.67), (157.51, 143.87), (158.37, 145.23), (159.0, 146.71), (159.38, 148.27), (159.51, 149.88)])
    data_base.add_polyline(f"k1","koleso",0, False, True, False)
    data_base.add_coordinates(f"k0",[(299.51, 149.88), (297.57, 173.94), (291.79, 197.38), (282.33, 219.59), (269.43, 239.99), (253.42, 258.06), (234.72, 273.33), (213.82, 285.4), (191.25, 293.96), (167.59, 298.78), (143.47, 299.76), (119.51, 296.85), (96.32, 290.13), (74.51, 279.78), (54.65, 266.07), (37.24, 249.35), (22.74, 230.05), (11.52, 208.67), (3.87, 185.78), (0.0, 161.95), (0.0, 137.81), (3.87, 113.98), (11.52, 91.08), (22.74, 69.71), (37.24, 50.41), (54.65, 33.69), (74.51, 19.97), (96.32, 9.63), (119.51, 2.91), (143.47, 0.0), (167.59, 0.97), (191.25, 5.8), (213.82, 14.36), (234.72, 26.43), (253.42, 41.7), (269.43, 59.77), (282.33, 80.17), (291.79, 102.38), (297.57, 125.82), (299.51, 149.88)])
    data_base.add_polyline(f"k0","koleso",0, False, True, False)


    data_base.add_coordinates(f"k2", [(137.53, 285.78), (137.53, 291.78), (161.5, 291.78), (161.5, 285.78), (137.53, 285.78)])
    data_base.add_polyline(f"k2","koleso",0, False, True, False)
    data_base.add_coordinates(f"k3", [(184.74, 281.68), (186.79, 287.32), (209.31, 279.13), (207.25, 273.49), (184.74, 281.68)])
    data_base.add_polyline(f"k3","koleso",0, False, True, False)
    data_base.add_coordinates(f"k4", [(227.69, 261.69), (231.55, 266.28), (249.91, 250.88), (246.05, 246.28), (227.69, 261.69)])
    data_base.add_polyline(f"k4","koleso",0, False, True, False)
    data_base.add_coordinates(f"k5", [(261.22, 228.21), (266.41, 231.21), (278.4, 210.45), (273.2, 207.45), (261.22, 228.21)])
    data_base.add_polyline(f"k5","koleso",0, False, True, False)
    data_base.add_coordinates(f"k6", [(281.27, 185.28), (287.18, 186.32), (291.34, 162.72), (285.43, 161.68), (281.27, 185.28)])
    data_base.add_polyline(f"k6","koleso",0, False, True, False)
    data_base.add_coordinates(f"k7", [(285.43, 138.08), (291.34, 137.04), (287.18, 113.44), (281.27, 114.48), (285.43, 138.08)])
    data_base.add_polyline(f"k7","koleso",0, False, True, False)
    data_base.add_coordinates(f"k8", [(273.2, 92.3), (278.4, 89.3), (266.41, 68.55), (261.22, 71.55), (273.2, 92.3)])
    data_base.add_polyline(f"k8","koleso",0, False, True, False)
    data_base.add_coordinates(f"k9", [(246.05, 53.47), (249.91, 48.88), (231.55, 33.47), (227.69, 38.07), (246.05, 53.47)])
    data_base.add_polyline(f"k9","koleso",0, False, True, False)
    data_base.add_coordinates(f"k10", [(207.25, 26.27), (209.31, 20.63), (186.79, 12.43), (184.74, 18.07), (207.25, 26.27)])
    data_base.add_polyline(f"k10","koleso",0, False, True, False)
    data_base.add_coordinates(f"k11", [(161.5, 13.97), (161.5, 7.97), (137.53, 7.97), (137.53, 13.97), (161.5, 13.97)])
    data_base.add_polyline(f"k11","koleso",0, False, True, False)
    data_base.add_coordinates(f"k12", [(114.29, 18.07), (112.24, 12.43), (89.72, 20.63), (91.77, 26.27), (114.29, 18.07)])
    data_base.add_polyline(f"k12","koleso",0, False, True, False)
    data_base.add_coordinates(f"k13", [(71.34, 38.07), (67.48, 33.47), (49.12, 48.88), (52.98, 53.47), (71.34, 38.07)])
    data_base.add_polyline(f"k13","koleso",0, False, True, False)
    data_base.add_coordinates(f"k14", [(37.81, 71.55), (32.61, 68.55), (20.63, 89.3), (25.83, 92.3), (37.81, 71.55)])
    data_base.add_polyline(f"k14","koleso",0, False, True, False)
    data_base.add_coordinates(f"k15", [(17.76, 114.48), (11.85, 113.44), (7.69, 137.04), (13.59, 138.08), (17.76, 114.48)])
    data_base.add_polyline(f"k15","koleso",0, False, True, False)
    data_base.add_coordinates(f"k16", [(13.59, 161.68), (7.69, 162.72), (11.85, 186.32), (17.76, 185.28), (13.59, 161.68)])
    data_base.add_polyline(f"k16","koleso",0, False, True, False)
    data_base.add_coordinates(f"k17", [(25.83, 207.45), (20.63, 210.45), (32.61, 231.21), (37.81, 228.21), (25.83, 207.45)])
    data_base.add_polyline(f"k17","koleso",0, False, True, False)
    data_base.add_coordinates(f"k18", [(52.98, 246.28), (49.12, 250.88), (67.48, 266.28), (71.34, 261.69), (52.98, 246.28)])
    data_base.add_polyline(f"k18","koleso",0, False, True, False)
    data_base.add_coordinates(f"k19", [(91.77, 273.49), (89.72, 279.13), (112.24, 287.32), (114.29, 281.68), (91.77, 273.49)])
    data_base.add_polyline(f"k19","koleso",0, False, True, False)
    



    


    data_base.add_coordinates(f"k21", [(0.0, 13.48), (1.5, 11.98), (6.0, 11.98), (6.0, 0.0), (166.0, 0.0), (166.0, 11.98), (170.5, 11.98), (172.0, 13.48), (172.0, 34.45), (170.5, 35.95), (166.0, 35.95), (166.0, 47.93), (6.0, 47.93), (6.0, 35.95), (1.5, 35.95), (0.0, 34.45), (0.0, 13.48)])
    data_base.add_polyline(f"k21","pol",0, False, True, False)










    redraw()

def GRBL_connect(sender):
    global laser 
    laser = LaserController(port=dpg.get_value('com_tag'))
    if (laser.connect()):
        dpg.bind_item_theme(sender, green_theme)
    else:
        dpg.bind_item_theme(sender, disabled_theme)


def kam_callback():
    












    for_buffer = 0.08
    for_buffer2 = 0.047 







    naruja = data_base.get_tag('color="2"')
    vnutr = data_base.get_tag('color="1"')
    vpol = []
    for vn in vnutr:
        vpol.append(Polygon(data_base.get_coordinates(vn)).buffer(-for_buffer))
    multipol = MultiPolygon(vpol)
    polygons = []
    for npp in naruja:
        pol = Polygon(data_base.get_coordinates(npp)).buffer(for_buffer)
        p = 0
        nnn = []
        for p in vpol:
            if shapely.contains(pol,p):
                nnn.append(p)
        for p in nnn:
            pol = shapely.difference(pol,p)
        # polygon2 = shapely.difference(pol,unary_union(shapely.intersection(pol,multipol)))
        polygon2 = pol
        if polygon2.geom_type == 'Polygon':
            polygons.append(polygon2)
            x1, y1 = polygon2.exterior.xy
            data_base.add_polyline(npp+f"__{p}",'nice_path2',1, False, True, False)
           
            data_base.add_coordinates(npp+f"__{p}",[(x_,y_) for x_,y_ in zip(x1,y1)])
            d = 0
            for p in polygon2.interiors:
                x1, y1 = p.xy
                data_base.add_polyline(npp+f"__{p}{d}",'nice_path2',1, False, True, False)
           
                data_base.add_coordinates(npp+f"__{p}{d}",[(x_,y_) for x_,y_ in zip(x1,y1)])
        else:
            for pol in polygon2.geoms:
                polygons.append(pol)
                x1, y1 = pol.exterior.xy
                data_base.add_polyline(npp+f"__{p}",'nice_path2',1, False, True, False)
                
                data_base.add_coordinates(npp+f"__{p}",[(x_,y_) for x_,y_ in zip(x1,y1)])
                p+=1
                d = 0
                for p in pol.interiors:
                    x1, y1 = p.xy
                    data_base.add_polyline(npp+f"__{p}_{d}",'nice_path2',1, False, True, False)
            
                    data_base.add_coordinates(npp+f"__{p}_{d}",[(x_,y_) for x_,y_ in zip(x1,y1)])

    dpg.add_button(label="nice_path",parent='butonss',tag="nice_path",callback=active_but)
    dpg.add_button(label="nice_path2",parent='butonss',tag="nice_path2",callback=active_but)
    width_lines = float(dpg.get_value('border_line_width'))
    lins = MultiLineString([((0, y), (110, y))for y in np.arange(0,100,width_lines)])

    linn = lins
    for p in polygons:
        linn = shapely.difference(linn,p)
    print(polygons[1].exterior.xy)
    # linn = shapely.difference(lins,polygons[0])
    c = 0
    
    for l in linn.geoms:
        coords = []
        
        coords.append((round(l.coords[0][0],4),  round(l.coords[0][1],4)))
        coords.append((round(l.coords[1][0],4), round(l.coords[1][1],4)))           
    
        data_base.add_polyline('nice_path'+f"{c}" ,'nice_path',0, False, True, False)
        data_base.add_coordinates('nice_path'+f"{c}",coords)
        c+=1
        
    redraw()

def nosik_callback():
    dpg.add_button(label="nosik",parent='butonss',tag="nosik",callback=active_but)
    dpg.add_button(label="nosik",parent='butonss',tag="nosik2",callback=active_but)
    points = [(45,15),(-1.2,15),(-1.2,0),(40,0)] + get_circle_points(center=(40,-5),radius=5,begin_angle=90,end_angle=0) + [(45,-6.2),(60,-6.2)]+ get_circle_points(center=(45,0),radius=15,begin_angle=0,end_angle=90) 
    
    data_base.add_coordinates(f"nosikk", points)
    data_base.add_polyline(f"nosikk","nosik",0, False, True, False)

    points2 = [(45,10),(0,10),(0,5),(45,5)] + get_circle_points(center=(45,0),radius=5,begin_angle=90,end_angle=0) + [(50,-5),(55,-5)]+ get_circle_points(center=(45,0),radius=10,begin_angle=0,end_angle=90) 
    
    data_base.add_coordinates(f"nosikkk", points2)
    data_base.add_polyline(f"nosikkk","nosik",0, False, True, False)


    points3 = [(45,11.5),(0,11.5),(0,3.5),(45,3.5)] + get_circle_points(center=(45,0),radius=3.5,begin_angle=90,end_angle=0) + [(48.5,-5),(56.5,-5)]+ get_circle_points(center=(45,0),radius=11.5,begin_angle=0,end_angle=90) 
    
    data_base.add_coordinates(f"nosikkkk", points3)
    data_base.add_polyline(f"nosikkkk","nosik2",0, False, True, False)

    redraw()
###########################################
##########################################
#############################################
def calculate_polyline_length(points):
    
    length = 0.0
    
    for i in range(len(points) - 1):
        # Координаты первой точки
        x1, y1 = points[i]
        # Координаты второй точки
        x2, y2 = points[i + 1]
        
        # Вычисление длины отрезка
        segment_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        length += segment_length
        
    return length
def test_callback():
    dpg.add_button(label="test",parent='butonss',tag="test1",callback=active_but)
    A = 39
    B = 43.83
    l = 73
    L = 106
    d = 12-0.424
    angle_degrees = 45.295

    y = ((d * l / np.tan(np.deg2rad(angle_degrees))) + (B * l))/(L-l)
    y = l * 360 / 2/np.pi/angle_degrees
    print(y)
   
    # radius = 193.11
    radius = y
    NUMLINS = 35
    NUMLINS2 = 3
    rect1 = [(0,radius),(-15,radius),(-15,radius+4),(-45,radius+4),(-45,radius),(-80,radius),(-80,radius+39),(-45,radius+39),(-45,radius+35),(-15,radius+35),(-15,radius+39),(0,radius+39)]


    data_base.add_coordinates(f"test", rect1)
    data_base.add_polyline(f"test","test1",0, False, True, False)
    
    
    

    angles2 = np.linspace(np.pi/2 - np.deg2rad(angle_degrees),np.pi/2, NUMLINS//2)
    angles3 = np.linspace(np.pi/2 - np.deg2rad(angle_degrees) + np.deg2rad(angle_degrees/(NUMLINS-3)) ,np.pi/2 - np.deg2rad(angle_degrees/(NUMLINS-3)), NUMLINS//2-1)

    points = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in reversed(angles2)]
    data_base.add_coordinates(f"rtest", points)
    data_base.add_polyline(f"rtest","test1",1, False, True, False)

    
   

    
    rect2 = [points[-1],(points[-1][0] + 12 * math.cos(np.deg2rad(angle_degrees)),points[-1][1] - 12 * math.sin(np.deg2rad(angle_degrees)))]
    rect2.append((rect2[-1][0]+ 4 * math.cos(np.deg2rad(angle_degrees- 90) ),rect2[-1][1]- 4 * math.sin(np.deg2rad(angle_degrees- 90))))
    rect2.append((rect2[-1][0]+ 18 * math.cos(np.deg2rad(angle_degrees) ),rect2[-1][1]- 18 * math.sin(np.deg2rad(angle_degrees))))
    rect2.append((rect2[-1][0]+ 4 * math.cos(np.deg2rad(angle_degrees+ 90) ),rect2[-1][1]- 4 * math.sin(np.deg2rad(angle_degrees+ 90))))
    rect2.append((rect2[-1][0]+ 10 * math.cos(np.deg2rad(angle_degrees) ),rect2[-1][1]- 10 * math.sin(np.deg2rad(angle_degrees))))
    rect2.append((rect2[-1][0]+ 43.83 * math.cos(np.deg2rad(angle_degrees- 90) ),rect2[-1][1]- 43.83 * math.sin(np.deg2rad(angle_degrees- 90))))
    rect2.append((rect2[-1][0]+ 10 * math.cos(np.deg2rad(angle_degrees - 180) ),rect2[-1][1]- 10 * math.sin(np.deg2rad(angle_degrees - 180 ))))
    rect2.append((rect2[-1][0]+ 4 * math.cos(np.deg2rad(angle_degrees+ 90) ),rect2[-1][1]- 4 * math.sin(np.deg2rad(angle_degrees+ 90))))
    rect2.append((rect2[-1][0]+ 18 * math.cos(np.deg2rad(angle_degrees- 180) ),rect2[-1][1]- 18 * math.sin(np.deg2rad(angle_degrees- 180))))
    rect2.append((rect2[-1][0]+ 4 * math.cos(np.deg2rad(angle_degrees- 90) ),rect2[-1][1]- 4 * math.sin(np.deg2rad(angle_degrees- 90))))
    rect2.append((rect2[-1][0]+ 0.424 * math.cos(np.deg2rad(angle_degrees- 180) ),rect2[-1][1]-  0.424 * math.sin(np.deg2rad(angle_degrees- 180))))


    data_base.add_coordinates(f"rrtest", rect2)
    data_base.add_polyline(f"rrtest","test1",0, False, True, False)
    center2 = [0,rect2[-1][1]- rect2[-1][0] * math.tan(np.deg2rad(-angle_degrees - 90 ))]
    
    points2 = [((radius + 39-center2[1]) * np.cos(angle) , (radius+ 39-center2[1]) * np.sin(angle) +center2[1] ) for angle in reversed(angles2)]
    data_base.add_coordinates(f"r2test", points2)
    data_base.add_polyline(f"r2test","test1",1, False, True, False)
    print(calculate_polyline_length(points2))
    print(calculate_polyline_length(points))
    points3 = [((radius + 10.3333333 - 0.264957 * center2[1]) * np.cos(angle) , (radius + 10.3333333- 0.264957 * center2[1]) * np.sin(angle) + 0.264957 *center2[1] ) for angle in reversed(angles2)]

    points4 = [((radius + 28.6666667- 0.735042 * center2[1]) * np.cos(angle) , (radius + 28.6666667- 0.735042 * center2[1]) * np.sin(angle) +  0.735042 *center2[1] ) for angle in reversed(angles2)]
 
    points5 = [((radius + 28.6666667 -4 - 0.632478 * center2[1]) * np.cos(angle) , (radius + 28.6666667 -4- 0.632478 * center2[1]) * np.sin(angle) +  0.632478 *center2[1] ) for angle in reversed(angles2)]
 
    points6 = [((radius + 10.3333333 + 4 - 0.3675213 * center2[1]) * np.cos(angle) , (radius + 10.3333333+ 4 - 0.3675213 * center2[1]) * np.sin(angle) + 0.3675213 *center2[1] ) for angle in reversed(angles2)]
  
    points7 = [((radius + 4 - 0.102564 * center2[1]) * np.cos(angle) , (radius + 4 - 0.102564 * center2[1]) * np.sin(angle) + 0.102564 *center2[1] ) for angle in reversed(angles3)]
   
    points8 = [((radius + 39-4- 0.897435 * center2[1]) * np.cos(angle) , (radius+ 39-4 - 0.897435 * center2[1]) * np.sin(angle) + 0.897435 *center2[1] ) for angle in reversed(angles3)]
   
    points9 = [((radius + 17.5 - 0.448717 * center2[1]) * np.cos(angle) , (radius + 17.5 - 0.448717 * center2[1]) * np.sin(angle) + 0.448717 *center2[1] ) for angle in reversed(angles3)]
    
    points10 = [((radius + 21.5 - 0.551282 * center2[1]) * np.cos(angle) , (radius + 21.5 - 0.551282 * center2[1]) * np.sin(angle) + 0.551282 *center2[1] ) for angle in reversed(angles3)]

    W1 = (-center2[1] - 4 * (NUMLINS2-1))/NUMLINS2
    
    for a,b in zip(points6,points5):
        data_base.add_coordinates(f"r2test{a}", [a,b])
        data_base.add_polyline(f"r2test{a}","test1",2, False, True, False)
    for a,b in zip(points,points3):
        data_base.add_coordinates(f"r2test{a}", [a,b])
        data_base.add_polyline(f"r2test{a}","test1",2, False, True, False)
    for a,b in zip(points2,points4):
        data_base.add_coordinates(f"r2test{a}", [a,b])
        data_base.add_polyline(f"r2test{a}","test1",2, False, True, False)
    for a,b in zip(points7,points9):
        data_base.add_coordinates(f"r2test{a}", [a,b])
        data_base.add_polyline(f"r2test{a}","test1",2, False, True, False)
    for a,b in zip(points8,points10):
        data_base.add_coordinates(f"r2test{a}", [a,b])
        data_base.add_polyline(f"r2test{a}","test1",2, False, True, False)
                  
    










    redraw()
    
####################################################
####################################################
####################################################

def esye():
    global esyedaflag
    fd.show_file_dialog()
    esyedaflag = True

def borderesye():
    global borderflag
    borderflag = True
    fd.show_file_dialog()
    
def n_callback():
    n = int(dpg.get_value('n_'))
    dpg.set_value('N_',n+1)

def N_callback():
    N = int(dpg.get_value('N_'))
    dpg.set_value('n_',N-1)
def Dzkk_callback():
    Dzkk = float(dpg.get_value('Dzkk'))
    N = int(dpg.get_value('N_'))
    dpg.set_value('Dkk',4*N*Dzkk/2/np.pi)
    dpg.set_value('e',Dzkk/4)    
def Dkk_callback():
    Dkk = float(dpg.get_value('Dkk'))
    N = int(dpg.get_value('N_'))
    dpg.set_value('Dzkk',np.pi*Dkk/2/N)
    dpg.set_value('e',np.pi*Dkk/8/N)   
def e_callback():
    e = float(dpg.get_value('e'))
    dpg.set_value('Dzkk',4*e)
    N = int(dpg.get_value('N_'))
    dpg.set_value('Dkk',8*e * N/np.pi)
dpg.create_context()
def set_place():
     global place_in_a_circle
     place_in_a_circle = True
X_AXIS_TAG = "x_axis_tag"
Y_AXIS_TAG = "y_axis_tag"

current_file = None
place_in_a_circle = False
poliline_themes = {}
esyedaflag = False
borderflag = False
data_base = PolylineDatabase()
laser = 'laser'


with dpg.window(label="Delete Files", show=False, tag="modal_id", no_title_bar=True):
    dpg.add_text("Layers")
    dpg.add_separator()
    
with dpg.window(label="Epitrohoida", show=False, tag="Epitrohoida_window", no_title_bar=True,pos=(400,100)):
    dpg.add_text("Epitrohoida")
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text("n")      
        dpg.add_input_text(width=50,scientific=True,tag='n_',default_value='50',callback=n_callback)
    with dpg.group(horizontal=True):
        dpg.add_text("N")      
        dpg.add_input_text(width=50,scientific=True,tag='N_',default_value='51',callback=N_callback) 

    with dpg.group(horizontal=True):
        dpg.add_text("Dzkk")      
        dpg.add_input_text(width=50,scientific=True,tag='Dzkk',default_value='2',callback=Dzkk_callback) 
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("Dkk")      
        dpg.add_input_text(width=50,scientific=True,tag='Dkk',default_value=f'{4*50*2/2/np.pi}',callback=Dkk_callback) 
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("e")
        dpg.add_input_text(width=50,scientific=True,tag='e',default_value='0.5',callback=e_callback) 
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("Count Points")
        dpg.add_input_text(width=50,scientific=True,tag='countp',default_value='1000') 
        
   


    with dpg.group(horizontal=True):
        dpg.add_spacer(width=50)
        dpg.add_button(label='Apply',callback=Epitrohoida_callback)
        dpg.add_spacer(width=50)

with dpg.window(label="Organizer", show=False, tag="Organizer_window", no_title_bar=True,pos=(400,100)):
    dpg.add_text("Organizer")
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text("Column count")      
        dpg.add_input_text(width=50,scientific=True,tag='Column',default_value='3')
        
    with dpg.group(horizontal=True):
        dpg.add_text("Row count")      
        dpg.add_input_text(width=50,scientific=True,tag='Row',default_value='3') 
    with dpg.group(horizontal=True):
        dpg.add_text("thickness")      
        dpg.add_input_text(width=50,scientific=True,tag='thickness',default_value='3') 
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("cell width")      
        dpg.add_input_text(width=50,scientific=True,tag='cell_width',default_value='60') 
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("cell height")      
        dpg.add_input_text(width=50,scientific=True,tag='cell_height',default_value='38') 
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("cell depth")      
        dpg.add_input_text(width=50,scientific=True,tag='cell_depth',default_value='100') 
        dpg.add_text("mm")
    



    with dpg.group(horizontal=True):
        dpg.add_spacer(width=50)
        dpg.add_button(label='Apply',callback=organizer_callback)
        dpg.add_spacer(width=50)


with dpg.window(label="EsyEDA", show=False, tag="border_from_esyeda", no_title_bar=True,pos=(400,100)):
    dpg.add_text("EsyEDA line")
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text("line width ")      
        dpg.add_input_text(width=50,scientific=True,tag='border_line_width',default_value='0.07')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("count lines")      
        dpg.add_input_text(width=50,scientific=True,tag='border_line_count',default_value='10') 
    
    dpg.add_separator()
    dpg.add_text("EsyEDA border")
    dpg.add_separator()
    dpg.add_checkbox(label='width border=0',tag='widthborder0',default_value=True)
    with dpg.group(horizontal=True):
        dpg.add_spacer(width=50)
        dpg.add_button(label='Apply',callback=lambda:dpg.configure_item("border_from_esyeda", show=False))
        dpg.add_spacer(width=50)
with dpg.window(label="Text Size", show=False, tag="text_size_modal", no_title_bar=True,pos=(400,100)):
    dpg.add_text("Text Size")
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text("height")      
        dpg.add_input_text(width=50,scientific=True,tag='text_size',default_value='10')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_spacer(width=50)
        dpg.add_button(label='Apply',callback=lambda:dpg.configure_item("text_size_modal", show=False))
        dpg.add_spacer(width=50)

with dpg.window(label="RECTANGLEFROMCENTER", show=False, tag="RECTANGLEFROMCENTER", no_title_bar=True,pos=(400,100)):
    dpg.add_text("RECTANGLE")
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text("center x")      
        dpg.add_input_text(width=50,scientific=True,tag='center xrec',default_value='0')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("center y")      
        dpg.add_input_text(width=50,scientific=True,tag='center yrec',default_value='0')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("width")      
        dpg.add_input_text(width=50,scientific=True,tag='widthrec',default_value='30')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("height")      
        dpg.add_input_text(width=50,scientific=True,tag='heightrec',default_value='10')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_spacer(width=50)
        dpg.add_button(label='Apply',callback=rectangle1_callback)
        dpg.add_spacer(width=50)

with dpg.window(label="CIRCLE", show=False, tag="CIRCLE", no_title_bar=True,pos=(400,100)):
    dpg.add_text("CIRCLE")
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text("radius")      
        dpg.add_input_text(width=50,scientific=True,tag='radius',default_value='10')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("center x")      
        dpg.add_input_text(width=50,scientific=True,tag='center x',default_value='0')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("center y")      
        dpg.add_input_text(width=50,scientific=True,tag='center y',default_value='0')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("start angle")      
        dpg.add_input_text(width=50,scientific=True,tag='start angle',default_value='0')
        
    with dpg.group(horizontal=True):
        dpg.add_text("end angle")      
        dpg.add_input_text(width=50,scientific=True,tag='end angle',default_value='360')
    with dpg.group(horizontal=True):
        dpg.add_text("num points")      
        dpg.add_input_text(width=50,scientific=True,tag='nps',default_value='40')
    
    with dpg.group(horizontal=True):
        dpg.add_spacer(width=50)
        dpg.add_button(label='Apply',callback=circle_callback)
        dpg.add_spacer(width=50)
    
with dpg.window(label="IZGIB", show=False, tag="IZGIB", no_title_bar=True,pos=(400,100)):
    dpg.add_text("IZGIB")
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text("a")      
        dpg.add_input_text(width=50,scientific=True,tag='a',default_value='0')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("b")      
        dpg.add_input_text(width=50,scientific=True,tag='b',default_value='238.754')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("c")      
        dpg.add_input_text(width=50,scientific=True,tag='c',default_value='0')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("d")      
        dpg.add_input_text(width=50,scientific=True,tag='d',default_value='50')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("number lines")      
        dpg.add_input_text(width=50,scientific=True,tag='numlins',default_value='6')
    with dpg.group(horizontal=True):
        dpg.add_text("width")      
        dpg.add_input_text(width=50,scientific=True,tag='w',default_value='4')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_spacer(width=50)
        dpg.add_button(label='Apply',callback=izgib_callback)
        dpg.add_spacer(width=50)

with dpg.window(label="KONUS", show=False, tag="KONUS", no_title_bar=True,pos=(400,100)):
    dpg.add_text("KONUS")
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text("l")      
        dpg.add_input_text(width=50,scientific=True,tag='l',default_value='36.652')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("L")      
        dpg.add_input_text(width=50,scientific=True,tag='L',default_value='57.596')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("d")      
        dpg.add_input_text(width=50,scientific=True,tag='d_',default_value='41.173')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("number points")      
        dpg.add_input_text(width=50,scientific=True,tag='numpoints',default_value='30')
    with dpg.group(horizontal=True):
        dpg.add_text("a")      
        dpg.add_input_text(width=50,scientific=True,tag='a_',default_value='40')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("w")      
        dpg.add_input_text(width=50,scientific=True,tag='w_',default_value='4')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("number lines")      
        dpg.add_input_text(width=50,scientific=True,tag='numlins_',default_value='4')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_spacer(width=50)
        dpg.add_button(label='Apply',callback=konus_callback)
        dpg.add_spacer(width=50)


with dpg.window(label="place_in_a_circle", show=False, tag="place_in_a_circle", no_title_bar=True,pos=(400,100)):
    dpg.add_text("place in a circle")
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_radio_button(parent="place_in_a_circle",items=['spacing','area'],tag='varrast',horizontal=True)
    with dpg.group(horizontal=True):
        dpg.add_text("center x")      
        dpg.add_input_text(width=50,scientific=True,tag='centerxplace',default_value='0')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("center y")      
        dpg.add_input_text(width=50,scientific=True,tag='centeryplace',default_value='0')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("count")      
        dpg.add_input_text(width=50,scientific=True,tag='countforspace',default_value='3')
        
    
    with dpg.group(horizontal=True):     
        dpg.add_input_text(width=50,scientific=True,tag='rastdegrees',default_value='30')
        dpg.add_text("degrees")
    dpg.add_text(tag='activetext')
    with dpg.group(horizontal=True):
        dpg.add_spacer(width=50)
        dpg.add_button(label='Apply',callback=place_in_a_circle_callback)
        dpg.add_spacer(width=50)









with dpg.theme(tag="plot_theme"):
        
        with dpg.theme_component(dpg.mvScatterSeries):
            dpg.add_theme_color(dpg.mvPlotCol_Line, (255, 0, 0), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Plus, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 6, category=dpg.mvThemeCat_Plots)


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
with dpg.theme() as green_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 255, 0), category=dpg.mvThemeCat_Core)


with dpg.theme() as disabled_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 0, 0), category=dpg.mvThemeCat_Core)

fd = FileDialog(callback=pr,width=800,height=400,filter_list=[".dxf",".png"])

#fd_esyeda = FileDialog(callback=esy_eda)

with dpg.viewport_menu_bar():
    with dpg.menu(label="File"):
        dpg.add_menu_item(label="Open", callback=fd.show_file_dialog)
        dpg.add_menu_item(label="Open DXF from EsyEDA", callback=esye)
        dpg.add_menu_item(label="Open Border from EsyEDA", callback=borderesye)
        dpg.add_menu_item(label="Save As Gcode", callback=save_as_gcode)
        dpg.add_menu_item(label="Save As DXF", callback=save_as_dxf)
        dpg.add_menu_item(label="Save Selected As DXF", callback=save_sel_as_dxf)
        
        with dpg.menu(label="Settings"):
            dpg.add_menu_item(label="Text Size", callback=lambda:dpg.configure_item("text_size_modal", show=True))
            dpg.add_menu_item(label="Border from EsyEDA", callback=lambda:dpg.configure_item("border_from_esyeda", show=True))
    with dpg.menu(label="Functions"):
        

        dpg.add_menu_item(label="Split selected", callback=split_l)
        dpg.add_menu_item(label="Normalize", callback=normalize_lines)
        dpg.add_menu_item(label="Move To Center", callback=move_to_center_lines)
        dpg.add_menu_item(label="Dublicate", callback=dublicate_lines)
        dpg.add_menu_item(label="Rotate Y", callback=rotate_x)
        dpg.add_menu_item(label="Rotate X", callback=rotate_y)
        dpg.add_menu_item(label="Delete", callback=delete_l)
        dpg.add_menu_item(label="Set Color", callback=set_color)
        dpg.add_menu_item(label="test", callback=test_callback)
        dpg.add_menu_item(label="Join", callback=join_callback)
        dpg.add_menu_item(label="Join selected (0.5mm)", callback=joinsel_callback)
        dpg.add_menu_item(label="Scale", callback=scale_lines)
        dpg.add_menu_item(label="buffer(-0.5)", callback=bufferm)
        dpg.add_menu_item(label="buffer(+0.5)", callback=bufferp)
        dpg.add_menu_item(label="Optimize", callback=optimize_)
        
        dpg.add_menu_item(label="Place in a circle", callback=lambda:(dpg.configure_item("place_in_a_circle", show=True),set_place()))
        dpg.add_menu_item(label="Print coords selected", callback=print_coords)
    with dpg.menu(label="Geom"):
        dpg.add_menu_item(label="Circle", callback=lambda:dpg.configure_item("CIRCLE", show=True))
        dpg.add_menu_item(label="Rectangle", callback=lambda:dpg.configure_item("RECTANGLEFROMCENTER", show=True))
        dpg.add_menu_item(label="Gears", callback=gears_callback)
        dpg.add_menu_item(label="Organizer", callback=lambda:dpg.configure_item("Organizer_window", show=True))
        dpg.add_menu_item(label="Epitrohoida", callback=lambda:dpg.configure_item("Epitrohoida_window", show=True))
        
    with dpg.menu(label="Generated"):
        with dpg.menu(label="test line"):
            dpg.add_menu_item(label="horizont line", callback=horizont_callback)
            dpg.add_menu_item(label="vertical line", callback=vertical_callback)
            dpg.add_menu_item(label="diagonal line", callback=diagonal_callback)
        with dpg.menu(label="izgib"):
            dpg.add_menu_item(label="konus", callback=lambda:dpg.configure_item("KONUS", show=True)) 
            dpg.add_menu_item(label="cilindr", callback=lambda:dpg.configure_item("IZGIB", show=True)) 

        dpg.add_menu_item(label="nosik", callback=nosik_callback) 
        
        dpg.add_menu_item(label="zapolnit dlya lazera", callback=kam_callback)
        dpg.add_menu_item(label="krysha nalivatora", callback=test2)
        
        dpg.add_menu_item(label="ruchka", callback=test10)
        dpg.add_menu_item(label="staraya versiya", callback=test4)
        dpg.add_menu_item(label="nalivator", callback=test5)
        dpg.add_menu_item(label="vlad", callback=testvlad1)
        dpg.add_menu_item(label="CNC bok", callback=test11)
        dpg.add_menu_item(label="koleso", callback=testkoleso)
        dpg.add_menu_item(label="sekciya", callback=test3)
        
    with dpg.menu(label="Widget Items"):
        dpg.add_checkbox(label="Pick Me", callback=print_me)
        dpg.add_button(label="Press Me", callback=print_me)
        dpg.add_color_picker(label="Color Me", callback=print_me)      
    with dpg.menu(label="G-Code"):
        dpg.add_menu_item(label="Load All", callback=load_all)
        dpg.add_menu_item(label="Load Selected", callback=load_sel)
        

with dpg.window(pos=(0,0),width=900, height=775,tag='papa'):
    
    with dpg.group(horizontal=True):
        with dpg.group():
            with dpg.file_dialog(directory_selector=False, show=False, callback=save_dxf, id="file_dialog_id1", width=700 ,height=400):
                    dpg.add_file_extension(".dxf", color=(255, 0, 255, 255), custom_text="[DXF]")
            with dpg.file_dialog(directory_selector=False, show=False, callback=save_sel_dxf, id="file_dialog_id11", width=700 ,height=400):
                    dpg.add_file_extension(".dxf", color=(255, 0, 255, 255), custom_text="[DXF]")
            with dpg.file_dialog(directory_selector=False, show=False, callback=callback_to_gcode2, id="file_dialog_id2", width=700 ,height=400):
                    dpg.add_file_extension(".gcode", color=(255, 0, 255, 255), custom_text="[GCODE]")

            
            with dpg.plot( width=600, height=600, tag="plot",no_menus=True, no_box_select=True) as plot:
                dpg.add_plot_axis(dpg.mvXAxis, label="X", tag=X_AXIS_TAG)
                
            
                yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Y", tag=Y_AXIS_TAG)
                dpg.add_scatter_series([0], [0], parent=Y_AXIS_TAG, tag="series_center")
                dpg.bind_item_theme("series_center", "plot_theme")
                # dpg.set_axis_limits_constraints(Y_AXIS_TAG,-80,210)
                # dpg.set_axis_limits_constraints(X_AXIS_TAG,-80,210)
                
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("order")
                    dpg.add_text("power")
                    dpg.add_text("speed")
                    dpg.add_text("repeat")
                with dpg.group():
                    dpg.add_checkbox(label="1",tag='color_1',callback=check_callback,default_value=True)
                    dpg.bind_item_theme(dpg.last_item(), coloured_Core_theme1)
                    dpg.add_input_text(width=50,scientific=True,tag='1_value',default_value='1000')
                    dpg.add_input_text(width=50,scientific=True,tag='11_value',default_value='120')
                    dpg.add_input_text(width=50,scientific=True,tag='111_value',default_value='1')
                with dpg.group():
                    dpg.add_checkbox(label="2",tag='color_2',callback=check_callback)
                    dpg.bind_item_theme(dpg.last_item(), coloured_Core_theme2)
                    dpg.add_input_text(width=50,scientific=True,tag='2_value',default_value='710')
                    dpg.add_input_text(width=50,scientific=True,tag='21_value',default_value='300')
                    dpg.add_input_text(width=50,scientific=True,tag='211_value',default_value='1')
                with dpg.group():
                    dpg.add_checkbox(label="3",tag='color_3',callback=check_callback)
                    dpg.bind_item_theme(dpg.last_item(), coloured_Core_theme3)
                    dpg.add_input_text(width=50,scientific=True,tag='3_value',default_value='1000')
                    dpg.add_input_text(width=50,scientific=True,tag='31_value',default_value='100')
                    dpg.add_input_text(width=50,scientific=True,tag='311_value',default_value='1')
                with dpg.group():
                    dpg.add_checkbox(label="4",tag='color_4',callback=check_callback)
                    dpg.bind_item_theme(dpg.last_item(), coloured_Core_theme4)
                    dpg.add_input_text(width=50,scientific=True,tag='4_value',default_value='1000')
                    dpg.add_input_text(width=50,scientific=True,tag='41_value',default_value='121')
                    dpg.add_input_text(width=50,scientific=True,tag='411_value',default_value='1')
                with dpg.group():
                    dpg.add_checkbox(label="5",tag='color_5',callback=check_callback)
                    dpg.bind_item_theme(dpg.last_item(), coloured_Core_theme5)
                    dpg.add_input_text(width=50,scientific=True,tag='5_value',default_value='1000')
                    dpg.add_input_text(width=50,scientific=True,tag='51_value',default_value='1000')
                    dpg.add_input_text(width=50,scientific=True,tag='511_value',default_value='1')
                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_spacer(width=20)
                        dpg.add_text("Center")
                        
                    
                    with dpg.group(horizontal=True):
                        with dpg.group():
                            dpg.add_text("X")
                            dpg.add_input_text(width=50,scientific=True,tag='x_center',default_value='0')   
                        with dpg.group(): 
                            dpg.add_text("Y")
                            dpg.add_input_text(width=50,scientific=True,tag='y_center',default_value='0')  

                    dpg.add_checkbox(label="edit",default_value=False,tag='center',callback=rasberitesb)
                dpg.add_checkbox(label="Split Polyline",default_value=False,tag='split',callback=rasberitesb)



        with dpg.group():
            dpg.add_spacer()
            dpg.add_spacer()
            dpg.add_spacer()
            dpg.add_text("completed")
            dpg.add_input_text(multiline=True, label="", default_value="", tag="multiline_input2",width=300,height=250)
            dpg.add_text("prepared")
            dpg.add_input_text(multiline=True, label="", default_value="", tag="multiline_input", readonly=False,width=300,height=250)
            

            dpg.add_checkbox(label="erase old",default_value=True,tag='eraseold')
            with dpg.group(horizontal=True):
                
                dpg.add_checkbox(label="paste numbers",default_value=False,tag='add_text',callback=rasberitesb)
                dpg.add_input_text(width=50,scientific=True,tag='insert_numbers',default_value='123')
            dpg.add_checkbox(label="change order",default_value=False,tag='change_order',callback=rasberitesb)
            dpg.add_checkbox(label="move lines",default_value=False,tag='movelines',callback=rasberitesb)
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="select",default_value=False,tag='select',callback=rasberitesb)
                dpg.add_button(label='select all',callback=calback_reselect,tag='selectall')
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="rotate",default_value=False,tag='rotate',callback=rasberitesb)
                dpg.add_input_text( label="", default_value="", tag="rotate_angle", readonly=False,scientific=True,width=50)
                dpg.add_button(label='rotate',tag='r',callback=call_rot)
        

with dpg.item_handler_registry() as registry:
    dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Right, callback=plot_mouse_click_callback)
dpg.bind_item_handler_registry(plot, registry)
dpg.set_primary_window("papa", True)

dpg.add_window(pos=(925,0),width=200, height=525,tag='butonss',label='lines')


with dpg.window(pos=(925,544),width=200, height=200,tag='pult',label=''):
    
        
    with dpg.group(horizontal=True,tag='forcombo'):
        dpg.add_button(label="check", callback=check_com_callback)
        dpg.add_button(label="load", callback=load_gcode_callback)
        dpg.add_combo(label="Port", items=['port','ne port'],width=60,tag='com_tag')
    with dpg.group(horizontal=True):
        dpg.add_button(label="Connect", callback=GRBL_connect)
    with dpg.group(horizontal=True):
        dpg.add_button(label='home',width=40,height=40,callback=calback_but1)
        dpg.add_button(label='^',width=40,height=40,callback=calback_but2)
        dpg.add_button(label='ver',width=40,height=40,callback=calback_but3)
    with dpg.group(horizontal=True):
        dpg.add_button(label='<',width=40,height=40,callback=calback_but4)
        dpg.add_button(label='0',width=40,height=40,callback=calback_but5)
        dpg.add_button(label='>',width=40,height=40,callback=calback_but6)
    with dpg.group(horizontal=True):
        dpg.add_button(label='hor',width=40,height=40,callback=calback_but7)
        dpg.add_button(label='v',width=40,height=40,callback=calback_but8)
        dpg.add_button(label='arc',width=40,height=40,callback=calback_but9)

dpg.create_viewport(width=1145, height=825, title="GCODE IDE")
dpg.setup_dearpygui()
dpg.show_viewport()

# test_callback()
dpg.start_dearpygui()

dpg.destroy_context()
