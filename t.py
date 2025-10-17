import sqlite3

class PolylineManager:
    def __init__(self, db_name):
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_tables()

    def create_tables(self):
        # Создание первой таблицы
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS polylines (
                tag TEXT PRIMARY KEY,
                color TEXT,
                active BOOLEAN,
                redraw_flag BOOLEAN,
                color_change_flag BOOLEAN
            )
        ''')
        
        # Создание второй таблицы
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS segments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tag TEXT,
                x_start REAL,
                y_start REAL,
                x_end REAL,
                y_end REAL,
                FOREIGN KEY (tag) REFERENCES polylines(tag)
            )
        ''')
        self.connection.commit()

    def add_polyline(self, tag, color, active, redraw_flag, color_change_flag):
        self.cursor.execute('''
            INSERT INTO polylines (tag, color, active, redraw_flag, color_change_flag)
            VALUES (?, ?, ?, ?, ?)
        ''', (tag, color, active, redraw_flag, color_change_flag))
        self.connection.commit()

    def add_segment(self, tag, x_start, y_start, x_end, y_end):
        self.cursor.execute('''
            INSERT INTO segments (tag, x_start, y_start, x_end, y_end)
            VALUES (?, ?, ?, ?, ?)
        ''', (tag, x_start, y_start, x_end, y_end))
        self.connection.commit()

    def get_polyline(self, tag):
        self.cursor.execute('SELECT * FROM polylines WHERE tag = ?', (tag,))
        return self.cursor.fetchone()

    def get_segments(self, tag):
        self.cursor.execute('SELECT * FROM segments WHERE tag = ?', (tag,))
        return self.cursor.fetchall()

    def update_polyline(self, tag, color=None, active=None, redraw_flag=None, color_change_flag=None):
        updates = []
        params = []

        if color is not None:
            updates.append("color = ?")
            params.append(color)
        if active is not None:
            updates.append("active = ?")
            params.append(active)
        if redraw_flag is not None:
            updates.append("redraw_flag = ?")
            params.append(redraw_flag)
        if color_change_flag is not None:
            updates.append("color_change_flag = ?")
            params.append(color_change_flag)

        if updates:
            params.append(tag)
            self.cursor.execute(f'''
                UPDATE polylines SET {', '.join(updates)} WHERE tag = ?
            ''', params)
            self.connection.commit()

    def close(self):
        self.connection.close()

# Пример использования
if __name__ == "__main__":
    pm = PolylineManager('polylines.db')
    pm.add_polyline('line1', 'red', True, False, False)
    pm.add_segment('line1', 0, 0, 1, 1)
    pm.add_segment('line1', 1, 1, 2, 3)
    print(pm.get_polyline('line1'))
    print(pm.get_segments('line1'))
    pm.update_polyline('line1', color='blue')
    pm.close()