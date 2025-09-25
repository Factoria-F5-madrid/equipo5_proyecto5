import psycopg2

conn = psycopg2.connect(database='healthdb', user='admin', password='admin', host = '192.168.0.130', port= '5432')
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FROM world_health;")
print(cur.fetchone())

cur.close()
conn.close()
