import pandas as pd
import MySQLdb 

db = MySQLdb.connect("localhost","otiose", "", "otiosedb")
query = "SELECT id, name, date_start FROM actor WHERE date_start IS NOT NULL"
actorTAble = pd.read_sql(query, db)

db.close()

print(actorTAble)
