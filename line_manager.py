import sqlite3
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import linemerge

class PolylineDatabase:
    def __init__(self, db_name='polylines.db'):
        self.connection = sqlite3.connect(db_name,check_same_thread=False)
        self.cursor = self.connection.cursor()
        self.drop_tables()
        self.create_tables()

    def create_tables(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS polylines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag TEXT UNIQUE,
                big_tag TEXT,
                color INTEGER,
                active BOOLEAN,
                redraw_flag BOOLEAN,
                color_change_flag BOOLEAN
            )
        ''')
        
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS coordinates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                polyline_tag TEXT,
                x REAL,
                y REAL,
                FOREIGN KEY (polyline_tag) REFERENCES polylines(tag)
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS photo (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag TEXT,
                x REAL,
                y REAL,
                color INTEGER
            )
        ''')
        self.connection.commit()

    def add_polyline(self, tag, big_tag, color, active, redraw_flag, color_change_flag):
        self.cursor.execute('''
            INSERT INTO polylines (tag, big_tag, color, active, redraw_flag, color_change_flag)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (tag, big_tag, color, active, redraw_flag, color_change_flag))
        self.connection.commit()
    def update_polylines(self, tags, **kwargs):
        if not tags:
            return

        columns = ', '.join(f"{key} = ?" for key in kwargs.keys())
        values = list(kwargs.values())
        
        # Создание строки для подстановки тегов
        placeholders = ', '.join('?' for _ in tags)
        values.extend(tags)

        self.cursor.execute(f'''
            UPDATE polylines
            SET {columns}
            WHERE tag IN ({placeholders})
        ''', values)
        self.connection.commit()
    def update_polyline(self, tag, **kwargs):
        columns = ', '.join(f"{key} = ?" for key in kwargs.keys())
        values = list(kwargs.values())
        values.append(tag)
        self.cursor.execute(f'''
            UPDATE polylines
            SET {columns}
            WHERE tag = ?
        ''', values)
        self.connection.commit()
    def get_unique_politag_where(self,condition):
        self.cursor.execute(f"SELECT DISTINCT big_tag FROM polylines WHERE {condition}")
        return [row[0] for row in self.cursor.fetchall()]
    def get_unique_politag(self):
        self.cursor.execute(f"SELECT DISTINCT big_tag FROM polylines")
        return [row[0] for row in self.cursor.fetchall()]
    def get_all_tag(self):
        self.cursor.execute(f'SELECT tag FROM polylines')
        return [row[0] for row in self.cursor.fetchall()]
    def get_tag_where(self, condition):
        self.cursor.execute(f'SELECT tag FROM polylines WHERE {condition}')
        return [row[0] for row in self.cursor.fetchall()]
    def get_bigtag_where(self, condition):
        self.cursor.execute(f'SELECT DISTINCT big_tag FROM polylines WHERE {condition}')
        return [row[0] for row in self.cursor.fetchall()]
    def get_polyline_where(self, condition):
        self.cursor.execute(f'SELECT * FROM polylines WHERE {condition}')
        return self.cursor.fetchall()
    def get_polylines_tag(self):
        self.cursor.execute(f'SELECT tag FROM polylines')
        return self.cursor.fetchall()
    def get_polyline(self, tag):
        self.cursor.execute('SELECT * FROM polylines WHERE tag = ?', (tag,))
        return self.cursor.fetchall()
    def get_tag(self,condition):
       
        self.cursor.execute(f'SELECT tag FROM polylines WHERE {condition}')
        return [row[0] for row in self.cursor.fetchall()]
    
    def fetch_tags_with_same_big_tag(self, tag_value):
        
        
        self.cursor.execute("""
        SELECT tag 
        FROM polylines 
        WHERE big_tag = (SELECT big_tag FROM polylines WHERE tag = ?);
        """, (tag_value,))
        tags = self.cursor.fetchall()
        
        
        return [tag[0] for tag in tags]

    def get_color(self,condition):
        
        self.cursor.execute(f'SELECT color, active FROM polylines WHERE {condition}')
        return self.cursor.fetchall()
    def add_coordinates(self, polyline_tag, coordinates):
        """
        Добавляет массив координат для заданной полилинии.

        :param polyline_tag: Тег полилинии.
        :param coordinates: Список кортежей с координатами (x, y).
        """
        data = [(polyline_tag, x, y) for x, y in coordinates]
        
        self.cursor.executemany('''
            INSERT INTO coordinates (polyline_tag, x, y)
            VALUES (?, ?, ?)
        ''', data)
        self.connection.commit()
    def get_all_coordinates(self):
        self.cursor.execute('SELECT * FROM coordinates ')
        return self.cursor.fetchall()
    def get_cord_by_tag(self):
        self.cursor.execute("SELECT polyline_tag , x, y FROM coordinates")
        return self.cursor.fetchall()
    
    def get_all_coordinates_where(self, condition):
        self.cursor.execute(f'SELECT * FROM coordinates WHERE {condition}')
        return self.cursor.fetchall()
    def get_coordinates(self, polyline_tag):
        self.cursor.execute('SELECT x, y FROM coordinates WHERE polyline_tag = ?', (polyline_tag,))
        return self.cursor.fetchall()
    def get_coordinates_with_id(self, polyline_tag):
        self.cursor.execute('SELECT id, x, y FROM coordinates WHERE polyline_tag = ?', (polyline_tag,))
        return self.cursor.fetchall()
    def update(self,update_queries):
        self.cursor.execute("BEGIN TRANSACTION;")
        self.cursor.executemany("UPDATE coordinates SET x = ?, y = ? WHERE id = ?;", update_queries)
        self.cursor.execute("COMMIT;")
    def xz(self, X,Y,polyline_tag):

        #distance_formula = f"((x - {X}) * (x - {X})) + ((y - {Y}) * (y - {Y}))"
        #placeholders = ', '.join('?' for i in range(len(polyline_tag)))
        # sql_query = f"""
        # SELECT 
        #     *, 
        #     {distance_formula} AS distance_sq
        # FROM coordinates WHERE polyline_tag IN ({placeholders})
        # ORDER BY 
        #     distance_sq ASC
        # LIMIT 1;
        # """
        
        placeholders = ', '.join(['?'] * len(polyline_tag))
    
        distance_formula = f"((x - {X}) * (x - {X})) + ((y - {Y}) * (y - {Y}))"
        
        sql_query = f"""
        WITH RankedPoints AS (
            SELECT
                *,
                ROW_NUMBER() OVER (PARTITION BY polyline_tag ORDER BY id ASC) AS rn,
                COUNT(*) OVER (PARTITION BY polyline_tag) AS max_rn
            FROM 
                coordinates
            WHERE
                polyline_tag IN ({placeholders})
        ),

        FirstAndLastPoints AS (
            SELECT 
                T.* FROM 
                coordinates T
            INNER JOIN 
                RankedPoints RP ON T.id = RP.id
            WHERE
                RP.rn = 1 OR RP.rn = RP.max_rn
        )

        SELECT 
            *, 
            {distance_formula} AS distance_sq
        FROM 
            FirstAndLastPoints
        ORDER BY 
            distance_sq ASC
        LIMIT 1;
        """



        self.cursor.execute(sql_query, polyline_tag)
        return self.cursor.fetchall()


        # placeholders = ', '.join('?' for i in range(len(polyline_tag)-2))
        # self.cursor.execute(f'SELECT x, y, polyline_tag, SQRT(POWER(x - ?, 2) + POWER(y - ?, 2)) AS distance FROM coordinates WHERE polyline_tag IN ({placeholders}) ORDER BY distance LIMIT 1', (polyline_tag,))
        # return self.cursor.fetchall()
        
    def get_coordinates_where(self, condition):
        self.cursor.execute(f'SELECT x, y FROM coordinates WHERE {condition}')
        return self.cursor.fetchall()
    def drop_tables(self):
        self.cursor.execute(f"DROP TABLE IF EXISTS coordinates")
        self.cursor.execute(f"DROP TABLE IF EXISTS polylines")
        self.connection.commit()
    def close(self):
        self.connection.close()
    def increment_field_value_with_condition(self,  delta_x, delta_y, condition):
        
        self.cursor.execute(f"""
            UPDATE coordinates
            SET x = x + ?,
            y = y + ?
            WHERE {condition}
        """, (delta_x, delta_y))
        self.connection.commit()
    def scale_value_with_condition(self,  delta_x, delta_y, condition):
        
        self.cursor.execute(f"""
            UPDATE coordinates
            SET x = x * ?,
            y = y * ?
            WHERE {condition}
        """, (delta_x, delta_y))
        self.connection.commit()

    def clear_tables(self):
        self.cursor.execute(f"DELETE FROM polylines")
        self.cursor.execute(f"DELETE FROM coordinates")
        self.connection.commit()
    def set_color(self, new_value):
        
        self.cursor.execute(f"UPDATE polylines SET color = ? WHERE active = 1", (new_value,))
        self.connection.commit()
    def select_all(self,):
        self.cursor.execute(f"UPDATE polylines SET active = True, color_change_flag = True WHERE active = False")
        self.connection.commit()
    def hide_all(self,):
        self.cursor.execute(f"UPDATE polylines SET active = False, color_change_flag = True WHERE active = True")
        self.connection.commit()
    def delete_active(self,condition):
        self.cursor.execute(f"DELETE FROM coordinates WHERE {condition}")
        self.cursor.execute(f"DELETE FROM polylines WHERE active=1")
        self.connection.commit()
    def delete(self,condition1,condition2):
        self.cursor.execute(f"DELETE FROM coordinates WHERE {condition1}")
        self.cursor.execute(f"DELETE FROM polylines WHERE {condition2}")
        self.connection.commit()
    def inverse_field_value_with_condition(self, column_name, increment_value, conditions):
      
        self.cursor.execute(f"""
            UPDATE coordinates
            SET {column_name} = ? - {column_name}
            WHERE {conditions}
        """, (increment_value,))
        self.connection.commit()
    def _extract_lines(self):
        # self.cursor.execute('''
        #     SELECT polyline_tag, x, y, id 
        #     FROM coordinates 
        #     ORDER BY polyline_tag, id
        # ''')
        self.cursor.execute('''
        SELECT c.polyline_tag, c.x, c.y, c.id 
        FROM coordinates AS c
        JOIN polylines AS p ON c.polyline_tag = p.tag
        WHERE p.active = True
        ORDER BY c.polyline_tag, c.id
        ''')
        all_data = self.cursor.fetchall()
        
        lines = []
        current_tag = None
        current_points = []
        
        for tag, x, y, order in all_data:
            
            if tag != current_tag and current_points:
                if len(current_points) >= 2:
                    lines.append(current_points)
                current_points = []
            
            current_tag = tag
            current_points.append((x, y))

        if current_points and len(current_points) >= 2:
            lines.append(current_points)
            
        return lines
    def merge_polylines(self,):
       
        try:        
            self.cursor.execute('''
                SELECT polyline_tag, x, y, id 
                FROM coordinates 
                ORDER BY polyline_tag, id
            ''')
            all_data = self.cursor.fetchall()
            
            # Группировка точек в ломаные линии по polyline_tag
            lines_to_merge = []
            current_tag = None
            current_points = []
            
            for tag, x, y, order in all_data:
                
                if tag != current_tag and current_points:
                    if len(current_points) >= 2:
                        lines_to_merge.append(LineString(current_points))
                    current_points = []
                
                current_tag = tag
                current_points.append((x, y))

            if current_points and len(current_points) >= 2:
                lines_to_merge.append(LineString(current_points))
                
            if not lines_to_merge:
                print("Нет ломаных линий для слияния.")
                return []

            merged_geometry = linemerge(lines_to_merge)

            results = []
            if merged_geometry.geom_type == 'LineString':
                results.append(merged_geometry)
            elif merged_geometry.geom_type == 'MultiLineString':
                
                results.extend(merged_geometry.geoms)
            
            return results

        except sqlite3.Error as e:
            print(f"Ошибка SQLite: {e}")
            return []
        


