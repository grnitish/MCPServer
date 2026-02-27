import pyodbc
import os

# Manual test config
connection_string = (
    "DRIVER={ODBC Driver 18 for SQL Server};"
    "SERVER=blueserver.database.windows.net;"
    "DATABASE=BlueriverInital;"
    "UID=readonly_user;"
    "PWD=**;"# Put your actual password here for 1 min to test
    "Encrypt=yes;"
    "TrustServerCertificate=yes;"
)

try:
    print("Testing Driver 18...")
    conn = pyodbc.connect(connection_string)
    print("✅ SUCCESS! Your driver and credentials are correct.")
    cursor = conn.cursor()


    cursor.execute("SELECT * FROM dbo.Portfolio")
    for row in cursor.fetchall():
        print(row)
    conn.close()
except Exception as e:
    print("❌ Still failing.")
    print(f"Error: {e}")