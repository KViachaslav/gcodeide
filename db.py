import sqlite3

class SQLiteDatabase:
    def __init__(self, db_name):
        
        self.connection = sqlite3.connect(db_name,check_same_thread=False)
        self.cursor = self.connection.cursor()

        self.base = []

    
    def create_table(self, table_name, columns_with_types):
    

        columns_definition = 'id INTEGER PRIMARY KEY AUTOINCREMENT, ' + ', '.join([f"{col} {dtype}" for col, dtype in columns_with_types])
        self.cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_definition})")
        self.connection.commit()
    def add_record(self, table_name, values):
        
        
        placeholders = ', '.join(['?' for _ in values])
        self.cursor.execute(f"INSERT INTO {table_name} (sx, sy, ex, ey, color, parent, isactive, forredraw) VALUES ({placeholders})", values)
        self.connection.commit()
    def add_multiple_records(self, table_name, records):
        """Добавление множества записей в таблицу."""
        placeholders = ', '.join('?' for _ in records[0])  # Создание плейсхолдеров для значений
        self.cursor.executemany(f"INSERT INTO {table_name} (sx, sy, ex, ey, color, parent, isactive, forredraw) VALUES ({placeholders})", records)
        self.connection.commit()
    def get_records(self, table_name):
        
        self.cursor.execute(f"SELECT * FROM {table_name}")
        return self.cursor.fetchall()
    
    def get_records_where(self, table_name, condition):
        
        self.cursor.execute(f"SELECT * FROM {table_name} WHERE {condition}")
        return self.cursor.fetchall()
    def update_record(self, table_name, record_id, column_name, new_value):
        """Обновление значения поля в таблице по id."""
        self.cursor.execute(f"UPDATE {table_name} SET {column_name} = ? WHERE id = ?", (new_value, record_id))
        self.connection.commit()
    def update_multiple_records(self, table_name, column_name, ids, value):
        
        ids_placeholder = ', '.join('?' for _ in ids)  # создание плейсхолдеров для SQL запроса
        self.cursor.execute(f"UPDATE lines SET {column_name} = {value} WHERE id IN ({ids_placeholder})", ids)
        
        self.connection.commit()
    def update_multiple_fields(self, table_name, toggle_column, column, target_value):
       
        self.cursor.execute(f"""
            UPDATE {table_name}
            SET {toggle_column} = CASE
                WHEN {toggle_column} = 0 THEN 1
                WHEN {toggle_column} = 1 THEN 0
                ELSE {toggle_column}  -- Сохраняем текущее значение для остальных случаев
            END,
            forredraw = 1
            WHERE {column} = ?
        """, (target_value,))
        self.connection.commit()
    def increment_field_value_with_condition(self, table_name, column_name1,column_name2,column_name3,column_name4, increment_value1,increment_value2, condition_column, condition_value):
      
        self.cursor.execute(f"""
            UPDATE {table_name}
            SET {column_name1} = {column_name1} + ?,
            {column_name2} = {column_name2} + ?,
            {column_name3} = {column_name3} + ?,
            {column_name4} = {column_name4} + ?,
            forredraw = 1
            WHERE {condition_column} = ?
        """, (increment_value1,increment_value1,increment_value2,increment_value2, condition_value))
        self.connection.commit()

    def inverse_field_value_with_condition(self, table_name, column_name1,column_name2, increment_value1, condition_column, condition_value):
      
        self.cursor.execute(f"""
            UPDATE {table_name}
            SET {column_name1} = ? - {column_name1},
            {column_name2} = ? - {column_name2},
            forredraw = 1
            WHERE {condition_column} = ?
        """, (increment_value1,increment_value1, condition_value))
        self.connection.commit()
    def set_color(self, table_name, new_value):
        
        self.cursor.execute(f"UPDATE {table_name} SET color = ? WHERE isactive = 1", (new_value,))
        self.connection.commit()
    def set_color_where_id(self, table_name, value,ids):
        ids_placeholder = ', '.join('?' for _ in ids)  # создание плейсхолдеров для SQL запроса

        self.cursor.execute(f"UPDATE {table_name} SET color = {value}, forredraw = 1 WHERE id IN ({ids_placeholder})", ids)
        self.connection.commit()
    def delete_record(self, table_name, condition):
        
        self.cursor.execute(f"DELETE FROM {table_name} WHERE {condition}")
        self.connection.commit()
    def delete_active(self, table_name):
        
        self.cursor.execute(f"DELETE FROM {table_name} WHERE isactive=1")
        self.connection.commit()
    def clear_table(self, table_name):
        
        self.cursor.execute(f"DELETE FROM {table_name}")
        self.connection.commit()
    def drop_table(self, table_name):
        """Удаление указанной таблицы из базы данных."""
        self.cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        self.connection.commit()
    def get_id_by_field(self, table_name, field_name, field_value):
        
        self.cursor.execute(f"""
            SELECT id FROM {table_name}
            WHERE {field_name} = ?
        """, (field_value,))
        return [row[0] for row in self.cursor.fetchall()]
    def close(self):
        
        self.connection.close()
    def get_parent_by_field_unique(self, table_name, field_name, field_value):
        
        self.cursor.execute(f"""
            SELECT DISTINCT parent FROM {table_name}
            WHERE {field_name} = ?
        """, (field_value,))
        return [row[0] for row in self.cursor.fetchall()]
    def get_unique_values(self, table_name, column_name):
        
        self.cursor.execute(f"SELECT DISTINCT {column_name} FROM {table_name}")
        return [row[0] for row in self.cursor.fetchall()]
    
if __name__ == "__main__":
    db = SQLiteDatabase('example.db')
    db.drop_table('lines')
    db.create_table('lines', [('sx', 'REAL'), ('sy', 'REAL'), ('ex', 'REAL'), ('ey', 'REAL'),('color', 'INTEGER'), ('parent', 'TEXT'),('isactive', 'INTEGER'),('forredraw', 'INTEGER')])

    db.add_record('lines', (1,  2, 3,4,0,'nice_path',0,1))
    print(db.get_records('lines'))
    db.close()