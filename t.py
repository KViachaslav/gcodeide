import sqlite3

def find_nearest_among_first_and_last(conn, X, Y):
   
    # Формула для вычисления квадрата Евклидова расстояния
    distance_formula = f"((x - {X}) * (x - {X})) + ((y - {Y}) * (y - {Y}))"
    
    # SQL-запрос с использованием CTE (Common Table Expressions)
    sql_query = f"""
    
    WITH RankedPoints AS (
        SELECT
            *,
            ROW_NUMBER() OVER (PARTITION BY tag ORDER BY id ASC) AS rn,
            COUNT(*) OVER (PARTITION BY tag) AS max_rn
        FROM coordinates
    ),

   
    FirstAndLastPoints AS (
        SELECT T.*
        FROM coordinates T
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
    
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        
        result = cursor.fetchone()
        return result

    except sqlite3.Error as e:
        print(f"Произошла ошибка SQLite: {e}")
        return None