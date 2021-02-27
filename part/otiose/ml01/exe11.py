import pandas as pd
import MySQLdb 

db = MySQLdb.connect("localhost","otiose", "", "otiosedb")
query = """SELECT actor.id, actor.date_start, actor.name, role.name, place.name 
FROM actor_has_role_and_place 
INNER JOIN actor ON actor.id = actor_has_role_and_place.actor
INNER JOIN role ON role.id = actor_has_role_and_place.role
INNER JOIN place ON place.id = actor_has_role_and_place.place
WHERE actor.date_start IS NOT NULL"""
actorTAble = pd.read_sql(query, db)

db.close()

print(actorTAble)
