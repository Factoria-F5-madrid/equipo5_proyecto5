import psycopg2

conn = psycopg2.connect(database='healthdb', user='admin', password='admin')
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM world_health;")
print(cur.fetchone())

cur.close()
conn.close()
