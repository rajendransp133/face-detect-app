import sqlite3

def insert_employee(db_name, name, photo_path, photo_path2,designations,hindi_name=None,tamil_name=None):
    conn = None
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        insert_query = """INSERT INTO employees
                        (name, photo_path, photo_path2,hindi_name,tamil_name,designations) VALUES (?, ?, ?,?,?,?)"""
        
        data_tuple = (name, photo_path, photo_path2,hindi_name,tamil_name,designations)
        cursor.execute(insert_query, data_tuple)
        conn.commit()
        cursor.close()
        return True

    except sqlite3.Error as error:
        print("Failed to insert employee data:", error)
        raise
    finally:
        if conn:
            conn.close()

def get_all_employees(db_name):
    conn = None
    employees = []
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        query = "SELECT id, name, photo_path, photo_path2,hindi_name,tamil_name,designations FROM employees"
        cursor.execute(query)
        employees = cursor.fetchall()
        
        cursor.close()
        
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if conn:
            conn.close()
            
    return employees

def get_employee(db_name, emp_id):
    conn = None
    employee = None
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        query = "SELECT id, name, photo_path, photo_path2,hindi_name,tamil_name,designations FROM employees WHERE id = ?"
        cursor.execute(query, (emp_id,))
        employee = cursor.fetchone()
        
        cursor.close()
        
    except sqlite3.Error as error:
        print("Failed to read data from sqlite table", error)
    finally:
        if conn:
            conn.close()
            
    return employee

def delete_employee(db_name, emp_id):
    conn = None

    try:
        emp_id = int(emp_id)
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM employees WHERE id = ?", (emp_id,))
        conn.commit()
        cursor.close()  

        return True

    except sqlite3.Error as error:
        print("Error while deleting user:", error)
        return False
    finally:
        if conn:
            conn.close()

def delete_all_employees(db_name):
    conn = None

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM employees")
        conn.commit()
        cursor.close()

        return True

    except sqlite3.Error as error:
        print("Error while deleting all users:", error)
        return False
    finally:
        if conn:
            conn.close()
