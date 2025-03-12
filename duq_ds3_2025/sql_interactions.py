import pandas as pd
import sqlite3
import sqlalchemy



def create_dataset():
    """Create a new sqlite database"""
    query = """
    CREATE TABLE patents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title VARCHAR(250) NOT NULL
    );    
    """
    # Connection to the database, not the database itself
    conn = sqlite3.connect('data/patent_npi_db.sqlite')
    cursor = conn.cursor()
    cursor.execute(query)
    cursor.execute('SELECT sqlite_version();')
    record = cursor.fetchall()
    print(record)
    cursor.close() # ensures that the changes are implemented by the database

if __name__ == '__main__':
    create_dataset()

