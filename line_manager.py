import sqlite3

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
    def get_all_coordinates_where(self, condition):
        self.cursor.execute(f'SELECT * FROM coordinates WHERE {condition}')
        return self.cursor.fetchall()
    def get_coordinates(self, polyline_tag):
        self.cursor.execute('SELECT x, y FROM coordinates WHERE polyline_tag = ?', (polyline_tag,))
        return self.cursor.fetchall()
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

    def clear_tables(self):
        self.cursor.execute(f"DELETE FROM polylines")
        self.cursor.execute(f"DELETE FROM coordinates")
        self.connection.commit()
    def set_color(self, new_value):
        
        self.cursor.execute(f"UPDATE polylines SET color = ? WHERE active = 1", (new_value,))
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
# db = PolylineDatabase()

# db.add_polyline('polyline1','polylineee1', 1, True, True, False)

# db.update_polyline('polyline1', color=3, active=True)

# polyline = db.get_polyline('polyline1')
# print(polyline)
# coordinates = [
#         (0, 0),
#         (1, 1),
#         (2, 2)
#     ]
    
# db.add_coordinates('polyline1', coordinates)

# coordinates = db.get_coordinates('polyline1')
# print(coordinates)


# db.close()