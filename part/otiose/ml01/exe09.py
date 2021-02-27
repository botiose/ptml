import pandas as pd
import MySQLdb 

db = MySQLdb.connect("localhost","otiose", "", "otiosedb")
cursor = db.cursor()
cursor.execute("SHOW TABLES")

d = {}

for table in cursor.fetchall():
    d[table[0]] = pd.read_sql("SELECT * FROM " + table[0], db)

db.close()

print(d)
    



