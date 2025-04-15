import sqlite3


def create_database(db_name):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        print("✅ Successfully connected to SQLite")
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                photo_path TEXT NOT NULL,
                photo_path2 TEXT NOT NULL
            );
        '''
        cursor.execute(create_table_query)
        conn.commit()
        print("📁 SQLite table 'employees' created (if it didn't already exist)")

    except sqlite3.Error as error:
        print("❌ Error while creating a SQLite table:", error)

    finally:
        if conn:
            cursor.close()
            conn.close()
            print("🔌 SQLite connection is closed")
