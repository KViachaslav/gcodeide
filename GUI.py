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
from db import SQLiteDatabase
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
def active_but(sender,app_data):
    state = data_base.get_polyline_where(f"big_tag='{sender}'")
    

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

    c = 0
    for k in range(num_lines):

        tunion_polygon = []
        if union_polygon.geom_type == 'Polygon':
            tunion_polygon.append(union_polygon.buffer(width_lines,quad_segs=0))
            xm, ym = union_polygon.exterior.xy

            data_base.add_polyline(nice_path+f"{c}",nice_path,0, False, True, False)
            data_base.add_coordinates(nice_path+f"{c}",[(x_,y_) for x_,y_ in zip(xm,ym)])
            c+=1
            
            for inter in union_polygon.interiors:
                xm, ym = inter.xy
                data_base.add_polyline(nice_path+f"{c}",nice_path,0, False, True, False)
                data_base.add_coordinates(nice_path+f"{c}",[(x_,y_) for x_,y_ in zip(xm,ym)])
                c+=1
            
        else:
           

            for p in union_polygon.geoms:
                
                tunion_polygon.append(p.buffer(width_lines,quad_segs=0))
                
                xm, ym = p.exterior.xy
                data_base.add_polyline(nice_path+f"{c}",nice_path,0, False, True, False)
                data_base.add_coordinates(nice_path+f"{c}",[(x_,y_) for x_,y_ in zip(xm,ym)])
                c+=1
                for inter in p.interiors:
                    xm, ym = inter.xy
                    data_base.add_polyline(nice_path+f"{c}",nice_path,0, False, True, False)
                    data_base.add_coordinates(nice_path+f"{c}",[(x_,y_) for x_,y_ in zip(xm,ym)])
                    c+=1
            
        union_polygon = unary_union(MultiPolygon([p for p in tunion_polygon]))
            
def read_dxf_lines_from_esyeda(sender, app_data, user_data):

    for_buffer = 0.07

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
    for circle in msp.query('CIRCLE'):
        layer = circle.dxf.layer
        
        if layer in layers:
            center = circle.dxf.center    
            num_points = 10  
            radius = circle.dxf.radius + for_buffer
            polygons.append(Polygon([(center.x + radius * math.cos(2 * math.pi * i / num_points),center.y + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))

           
        
    for polyline in msp.query('LWPOLYLINE'):
        layer = polyline.dxf.layer
        if layer in layers:
            w = polyline.dxf.const_width
            points = polyline.get_points()  
        
            num_points = 10
            radius = w/2 + for_buffer
            polygons.append(Polygon([(points[0][0] + radius * math.cos(2 * math.pi * i / num_points),points[0][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))    
            for j in range(len(points) - 1):
                num_points = 10
                radius = w/2 + for_buffer
                boundaries = calculate_boundary_coordinates(points[j][0], points[j][1], points[j + 1][0], points[j + 1][1], w + for_buffer*2)
                polygons.append(Polygon([(points[j + 1][0] + radius * math.cos(2 * math.pi * i / num_points),points[j + 1][1] + radius * math.sin(2 * math.pi * i / num_points))for i in range(num_points)]))
                
                polygons.append(Polygon([(boundaries['left_start'][0],boundaries['left_start'][1]), (boundaries['left_end'][0],boundaries['left_end'][1]), (boundaries['right_end'][0],boundaries['right_end'][1]), (boundaries['right_start'][0],boundaries['right_start'][1])]))

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
                
    lins = []
    if full:
        ex = shapely.envelope(unary_union(MultiPolygon([p for p in border])))
        print(ex.xy[0],ex.xy[1])
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
            Polygon_to_lines(unary_union(MultiPolygon([p for p in polygons])),1,width_lines,nice_path+ '_border')
            
            
            dpg.add_button(label=nice_path + '_border',parent='butonss',tag=nice_path + '_border',callback=active_but)
            print(nice_path + '_border')
    else:
        print('not full')
        multipolygon = MultiPolygon([p for p in polygons])
        union_polygon = unary_union(multipolygon)
        Polygon_to_lines(union_polygon,num_lines,width_lines,nice_path)
        
    redraw()













def read_dxf_lines(file_path):
    nice_path = os.path.basename(file_path)
    iterr = 1
    while 1:
        for i in data_base.get_unique_politag():
            if i == nice_path:
                nice_path = os.path.basename(file_path) + f' (copy {iterr})'
                iterr +=1
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
            dpg.set_value(f"{h}_value",power)
            dpg.set_value(f"{h}1_value",speed)
            lll[layer.dxf.name] = h - 1 
            h+=1


    border_lines = []
    colors = []
    counter= 0
    for line in msp.query('LINE'):
        
        layer = line.dxf.layer
        #border_lines.append([(round(line.dxf.start.x,4),  round(line.dxf.start.y,4)), (round(line.dxf.end.x,4), round(line.dxf.end.y,4))])
        if layer in ll:
            colors.append(lll[layer])
            data_base.add_polyline(nice_path+f"_line_"+f"{counter}",nice_path,lll[layer], False, True, False)
        else:
            colors.append(0)
            data_base.add_polyline(nice_path+f"_line_"+f"{counter}",nice_path,0, False, True, False)
        data_base.add_coordinates(nice_path+f"_line_"+f"{counter}", [[round(line.dxf.start.x,4),  round(line.dxf.start.y,4)], [round(line.dxf.end.x,4), round(line.dxf.end.y,4)]])
        
        counter+=1
    for spline in msp.query('SPLINE'):
        layer = spline.dxf.layer
        
        control_points = list(spline.control_points)
        fit_points = list(spline.fit_points)
        
        spline_coords = []
        
        if len(fit_points)!= 0:
            spline_coords = [[round(p[0], 4), round(p[1], 4)] for p in fit_points]
        
        elif control_points:
            
            degree = spline.dxf.degree
            num_segments = max(10, len(control_points) * 5)  
            
            for i in range(num_segments + 1):
                t = i / num_segments
                try:
                    point = spline.evaluate_point(t)
                    spline_coords.append([round(point.x, 4), round(point.y, 4)])
                except:
                   
                    spline_coords = [[round(p[0], 4), round(p[1], 4)] for p in control_points]
                    break
        
        if not spline_coords and control_points:
            spline_coords = [[round(p[0], 4), round(p[1], 4)] for p in control_points]
        
        if len(spline_coords) > 1:
            if layer in ll:
                colors.append(lll[layer])
                data_base.add_polyline(nice_path + f"_spline_{counter}", nice_path, lll[layer], False, True, False)
            else:
                colors.append(0)
                data_base.add_polyline(nice_path + f"_spline_{counter}", nice_path, 0, False, True, False)
            
            data_base.add_coordinates(nice_path + f"_spline_{counter}", spline_coords)
            counter += 1



    # sett = {i for i in range(len(border_lines))}
    
    # counter = 0
    # while sett:
    #     print(len(sett))
    #     i = next(iter(sett))
    #     coords = []
    #     l,m = find_closest_lines(border_lines,border_lines[i][0],sett)
        
        
    #     coords.append((round(border_lines[i][0][0],4),  round(border_lines[i][0][1],4)))
    #     color = colors[i]
    #     for h,j in zip(l,m):
    #         if j:
    #             coords.append((round(border_lines[h][1][0],4),  round(border_lines[h][1][1],4)))
    #         else:
    #             coords.append((round(border_lines[h][0][0],4),  round(border_lines[h][0][1],4)))
    #         sett.remove(h)
    #     data_base.add_polyline(nice_path+f"_3dface_"+f"{counter}",nice_path,color, False, True, False)
    #     data_base.add_coordinates(nice_path+f"_3dface_"+f"{counter}", coords)
    #     counter+=1

    
        
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
    
    
    border_lines = []
    for i in settt:
        border_lines.append([(round(hlines[i]['start'][0],4),  round(hlines[i]['start'][1],4)),(round(hlines[i]['end'][0],4),  round(hlines[i]['end'][1],4))])


    sett = {i for i in range(len(border_lines))}
    
    counter = 0
    while sett:
        i = next(iter(sett))
        coords = []
        l,m = find_closest_lines(border_lines,border_lines[i][0],sett)
        
        
        coords.append((round(border_lines[i][0][0],4),  round(border_lines[i][0][1],4)))
       
        for h,j in zip(l,m):
            if j:
                coords.append((round(border_lines[h][1][0],4),  round(border_lines[h][1][1],4)))
            else:
                coords.append((round(border_lines[h][0][0],4),  round(border_lines[h][0][1],4)))
            sett.remove(h)
        data_base.add_polyline(nice_path+f"_3dface_"+f"{counter}",nice_path,0, False, True, False)
        data_base.add_coordinates(nice_path+f"_3dface_"+f"{counter}", coords)
        counter+=1








   
    counter = 0
    for arc in msp.query('ARC'):
        center = arc.dxf.center  
        radius = arc.dxf.radius   
        start_angle = arc.dxf.start_angle 
        end_angle = arc.dxf.end_angle
        layer = arc.dxf.layer
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


    for circle in msp.query('CIRCLE'):
        center = circle.dxf.center 
        radius = circle.dxf.radius  
        num_points = 50  
        layer = circle.dxf.layer
        points = [
            (
                center.x + radius * math.cos(2 * math.pi * i / num_points),
                center.y + radius * math.sin(2 * math.pi * i / num_points)
            )
            for i in list(range(num_points)) + [0]
        ]
        if layer in ll:
            data_base.add_polyline(nice_path+f"_circle_"+f"{counter}",nice_path,lll[layer], False, True, False)
        else:
            data_base.add_polyline(nice_path+f"_circle_"+f"{counter}",nice_path,0, False, True, False)
        data_base.add_coordinates(nice_path+f"_circle_"+f"{counter}", points)
        counter+=1
    # counter = 0
    # for polyline in msp.query('SOLID'):
    #     layer = polyline.dxf.layer
        
    #     points = polyline.get_points() 
    #     coords = []
    #     for i in range(len(points)):
    #         coords.append((round(points[i][0],4),  round(points[i][1],4)))
    #     if layer in ll:
    #         data_base.add_polyline(nice_path+f"_solid_"+f"{counter}",nice_path,lll[layer], False, True, False)
    #     else:
    #         data_base.add_polyline(nice_path+f"_solid_"+f"{counter}",nice_path,0, False, True, False)
    #     data_base.add_coordinates(nice_path+f"_solid_"+f"{counter}", coords)
    #     counter +=1
            
    counter= 0
    for polyline in msp.query('LWPOLYLINE'):
        layer = polyline.dxf.layer
        points = polyline.get_points() 
        coords = []
        for i in range(len(points)):
            coords.append((round(points[i][0],4),  round(points[i][1],4)))
        coords.append((round(points[0][0],4),  round(points[0][1],4)))
        if layer in ll:
            data_base.add_polyline(nice_path+f"_lwpoly_"+f"{counter}",nice_path,lll[layer], False, True, False)
        else:
            data_base.add_polyline(nice_path+f"_lwpoly_"+f"{counter}",nice_path,0, False, True, False)
        data_base.add_coordinates(nice_path+f"_lwpoly_"+f"{counter}", coords)
        counter +=1
    
    counter= 0
    for polyline in msp.query('POLYLINE'):
        layer = polyline.dxf.layer
        points = polyline.vertices
        
        if layer in ll:
            data_base.add_polyline(nice_path+f"_poly_"+f"{counter}",nice_path,lll[layer], False, True, False)
        else:
            data_base.add_polyline(nice_path+f"_poly_"+f"{counter}",nice_path,0, False, True, False)
        data_base.add_coordinates(nice_path+f"_poly_"+f"{counter}", [(p.dxf.location.x,p.dxf.location.y)for p in points])
        counter +=1
    counter= 0
    for hatch in msp.query('HATCH'):
        for path in hatch.paths:
            layer = hatch.dxf.layer
            
            coords = []
            for i in range(len(points)):
                coords.append((round(points[i][0],4),  round(points[i][1],4)))
            if layer in ll:
                data_base.add_polyline(nice_path+f"_hatch_"+f"{counter}",nice_path,lll[layer], False, True, False)
            else:
                data_base.add_polyline(nice_path+f"_hatch_"+f"{counter}",nice_path,0, False, True, False)
            data_base.add_coordinates(nice_path+f"_hatch_"+f"{counter}", coords)
            counter +=1
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
def callback_to_gcode(sender, app_data, user_data):
    current_file = app_data['file_path_name']
    gcode_lines = []
    gcode_lines.append("G90")
    gcode_lines.append("M4 S0")

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
        h+=1
        for t in tag:
            coords = data_base.get_coordinates(t)

            gcode_lines.append(f"G0 X{round(coords[0][0],4)} Y{round(coords[0][1],4)}")
            gcode_lines.append(f"F{speed}")
            gcode_lines.append(f"S{power}")
            gcode_lines.append(f"G1 X{round(coords[1][0],4)} Y{round(coords[1][1],4)}")
            if len(coords) > 2:
                for coord in coords[2:]:
                    gcode_lines.append(f"X{round(coord[0],4)} Y{round(coord[1],4)}")
            gcode_lines.append("S0")

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
        # rec = db.get_records_where('lines','isactive=1')
        # xx = [r[1] for r in rec]
        # xx+= [r[3] for r in rec]
        # yy = [r[2] for r in rec]
        # yy+= [r[4] for r in rec]
        # db.increment_field_value_with_condition('lines','sx','ex','sy','ey',x-min(xx),y-min(yy),'isactive',1)
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
    
def redraw(all=0):

    global poliline_themes
    

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
        poliline_themes[f'{tag}'] = coloured_line_theme1
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

def optimize_():
    
    return
    #optimize.create_continuous_lines('temp.dxf',lines )

def rotate_x():
    invers_lines()
def rotate_y():
    invers_lines('y')
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

def extract_points_from_ggb(ggb_file_path):
    nice_path = find_nice_path(ggb_file_path)
    dpg.add_button(label=nice_path,parent='butonss',tag=nice_path,callback=active_but)
    points = {}
    segments = [] 
    
    try:
        with zipfile.ZipFile(ggb_file_path, 'r') as zip_ref:
            xml_content = zip_ref.read('geogebra.xml').decode('utf-8')
            print(xml_content)
    except Exception as e:
        print(f"Ошибка при чтении ZIP: {e}")
        return points, segments
    
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Ошибка парсинга XML: {e}")
        return points, segments
    
    for elem in root.iter('element'):
        if elem.get('type') == 'point':
            label = elem.get('label', 'unnamed')
            coords_elem = elem.find('coords')  # <coords x="..." y="..." z="..."/>
            if coords_elem is not None:
                x = float(coords_elem.get('x', 0.0))
                y = float(coords_elem.get('y', 0.0))
                points[label] = (x, y)
    counter = 0
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
                
            start_angle = math.atan2(points[a0][1] - c[1], points[a0][0] - c[0])
            end_angle = math.atan2(points[a2][1] - c[1], points[a2][0] - c[0])

            angles = [start_angle + ( i * (end_angle - start_angle)/num_points) for i in range(num_points)]
            
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
            print(cmd_name)    
        counter+=1            
    
    return points, segments



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
        data_base.clear_tables()
        dpg.delete_item(Y_AXIS_TAG, children_only=True, slot=1)
    if '.dxf' in current_file: 
        

        if esyedaflag:
            esyedaflag = False
            normicks = ['TopLayer','BoardOutLine','Multi-Layer']
            doc = ezdxf.readfile(current_file)
            layers = doc.layers
            print(layers)
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
            read_dxf_lines(current_file)
          
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

        
def check_com_callback():
    ports = serial.tools.list_ports.comports()
    dpg.delete_item('com_tag')
    dpg.add_combo(label="Port", items=[port.device for port in ports],width=60,tag='com_tag',parent='forcombo')
    
def load_gcode_callback():


    PORT = dpg.get_value('com_tag') 
    BAUDRATE = 115200


    gcode_commands = dpg.get_value('multiline_input').split('\n')
    print(gcode_commands)
    try:
       
        ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        time.sleep(2)  

        ser.write(b'\r\n\r\n')
        time.sleep(2)
        ser.flushInput() 

        print("Подключено к GRBL.")

        for line in gcode_commands:
            if line != '':
                l = line.strip()
                print(f"Отправка: {l}")
                ser.write((l + '\n').encode('utf-8')) 

                grbl_out = ser.readline().decode('utf-8').strip()
                print(f"Ответ GRBL: {grbl_out}")

        print("Отправка G-code завершена.")

    except serial.SerialException as e:
        print(f"Ошибка последовательного порта: {e}")

    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Порт закрыт.")


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
    data_base.add_coordinates(f"1", points + [(points[len(points)-1][0],points[len(points)-1][1]+a),((points2[0][0],points2[0][1]+a))] + points2 + [calculate_point(points2[len(points)-1],-90 + angle_degrees,a),calculate_point(points[0],-90 + angle_degrees,a),points[0]])
    data_base.add_polyline(f"1","arc",0, False, True, False)
    
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
    print(round(10/w))
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
    print(round(10/w))
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
###########################################
##########################################
#############################################
def test_callback():
    dpg.add_button(label="col",parent='butonss',tag="col",callback=active_but)
    data_base.add_coordinates(f"5", [(0,0),(197,0),(197,120),(0,120),(0,0)])
    data_base.add_polyline(f"5","col",0, False, True, False)
    h = 22.5
    for col in range(39):
        data_base.add_polyline(f"{col}1","col",0, False, True, False)
        data_base.add_polyline(f"{col}2","col",0, False, True, False)
        data_base.add_polyline(f"{col}3","col",0, False, True, False)
        data_base.add_polyline(f"{col}4","col",0, False, True, False)
        data_base.add_coordinates(f"{col}1", [(col*4+h,0),(col*4+h,18)])
        data_base.add_coordinates(f"{col}2", [(col*4+h,22),(col*4+h,58)])
        data_base.add_coordinates(f"{col}3", [(col*4+h,62),(col*4+h,98)])
        data_base.add_coordinates(f"{col}4", [(col*4+h,102),(col*4+h,120)])
        data_base.add_polyline(f"{col}5","col",0, False, True, False)
        data_base.add_polyline(f"{col}6","col",0, False, True, False)
        data_base.add_polyline(f"{col}7","col",0, False, True, False)
        
        data_base.add_coordinates(f"{col}5", [(col*4+2+h,4),(col*4+2+h,38)])
        data_base.add_coordinates(f"{col}6", [(col*4+2+h,42),(col*4+2+h,77)])
        data_base.add_coordinates(f"{col}7", [(col*4+2+h,82),(col*4+2+h,116)])
        



    redraw()
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

poliline_themes = {}
esyedaflag = False

data_base = PolylineDatabase()



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


with dpg.window(label="IZGIB", show=False, tag="IZGIB", no_title_bar=True,pos=(400,100)):
    dpg.add_text("IZGIB")
    dpg.add_separator()
    with dpg.group(horizontal=True):
        dpg.add_text("a")      
        dpg.add_input_text(width=50,scientific=True,tag='a',default_value='10')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("b")      
        dpg.add_input_text(width=50,scientific=True,tag='b',default_value='157')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("c")      
        dpg.add_input_text(width=50,scientific=True,tag='c',default_value='10')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("d")      
        dpg.add_input_text(width=50,scientific=True,tag='d',default_value='100')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("number lines")      
        dpg.add_input_text(width=50,scientific=True,tag='numlins',default_value='3')
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
        dpg.add_input_text(width=50,scientific=True,tag='l',default_value='103.6')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("L")      
        dpg.add_input_text(width=50,scientific=True,tag='L',default_value='157')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("d")      
        dpg.add_input_text(width=50,scientific=True,tag='d_',default_value='48')
        dpg.add_text("mm")
    with dpg.group(horizontal=True):
        dpg.add_text("number points")      
        dpg.add_input_text(width=50,scientific=True,tag='numpoints',default_value='45')
    with dpg.group(horizontal=True):
        dpg.add_text("a")      
        dpg.add_input_text(width=50,scientific=True,tag='a_',default_value='10')
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
            dpg.add_menu_item(label="Text Size", callback=lambda:dpg.configure_item("text_size_modal", show=True))
            dpg.add_menu_item(label="Border from EsyEDA", callback=lambda:dpg.configure_item("border_from_esyeda", show=True))
    with dpg.menu(label="Functions"):
        

        dpg.add_menu_item(label="Split", callback=split_l)
        dpg.add_menu_item(label="Normalize", callback=normalize_lines)
        dpg.add_menu_item(label="Rotate X", callback=rotate_x)
        dpg.add_menu_item(label="Rotate Y", callback=rotate_y)
        dpg.add_menu_item(label="Delete", callback=delete_l)
        dpg.add_menu_item(label="Set Color", callback=set_color)
        dpg.add_menu_item(label="test", callback=test_callback)
          


    with dpg.menu(label="Generated"):
        dpg.add_menu_item(label="horizont line", callback=horizont_callback)
        dpg.add_menu_item(label="vertical line", callback=vertical_callback)
        dpg.add_menu_item(label="diagonal line", callback=diagonal_callback)
        dpg.add_menu_item(label="konus", callback=lambda:dpg.configure_item("KONUS", show=True)) 
        dpg.add_menu_item(label="izgib", callback=lambda:dpg.configure_item("IZGIB", show=True)) 

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

            
            with dpg.plot( width=600, height=600, tag="plot",no_menus=True, no_box_select=True) as plot:
                dpg.add_plot_axis(dpg.mvXAxis, label="X", tag=X_AXIS_TAG)
                
            
                yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Y", tag=Y_AXIS_TAG)
                
                # dpg.set_axis_limits_constraints(Y_AXIS_TAG,-10,310)
                # dpg.set_axis_limits_constraints(X_AXIS_TAG,-10,310)
            
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
            dpg.add_input_text(multiline=True, label="", default_value="", tag="multiline_input", readonly=False,width=300,height=600)
            dpg.add_checkbox(label="erase old",default_value=True,tag='eraseold')
            with dpg.group(horizontal=True):
                
                dpg.add_checkbox(label="paste numbers",default_value=False,tag='add_text',callback=rasberitesb)
                dpg.add_input_text(width=50,scientific=True,tag='insert_numbers',default_value='123')
            dpg.add_checkbox(label="change order",default_value=False,tag='change_order',callback=rasberitesb)
            dpg.add_checkbox(label="move lines",default_value=False,tag='movelines',callback=rasberitesb)
    
with dpg.item_handler_registry() as registry:
    dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Right, callback=plot_mouse_click_callback)
dpg.bind_item_handler_registry(plot, registry)


dpg.add_window(pos=(900,0),width=200, height=525,tag='butonss',label='lines')


with dpg.window(pos=(900,544),width=200, height=200,tag='pult',label=''):
   
        
    with dpg.group(horizontal=True,tag='forcombo'):
        dpg.add_button(label="check", callback=check_com_callback)
        dpg.add_button(label="load", callback=load_gcode_callback)
        dpg.add_combo(label="Port", items=['port','ne port'],width=60,tag='com_tag')
    
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
dpg.create_viewport(width=1115, height=785, title="GCODE IDE")
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()